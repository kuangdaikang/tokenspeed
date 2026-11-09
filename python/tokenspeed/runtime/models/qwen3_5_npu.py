# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Qwen3.5 text adapter for Ascend, including eager TP.

Reuses Transformers' text decoder, RoPE, and per-request DynamicCache.
GDN, convolution, attention, and the dense MLP follow the column/row split
used by ``qwen3_5.py``: each logical projection is sharded on its own output
or input groups, and the row-parallel partials are summed with HCCL.
Embedding and the tied LM head stay replicated. This runner is separate from
the serving scheduler.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from tokenspeed_kernel.ops.attention.qwen35.ascend import (
    causal_conv,
    gdn_chunk,
    gdn_recurrent,
    mha_cached,
    rmsnorm,
)
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5RMSNormGated,
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
)

from tokenspeed.runtime.models.qwen3_5_npu_parallel import (
    TPConfig,
    allreduce_sum,
    current,
    grouped_slice,
    install,
    project_and_reduce,
    require_divisible,
)


class NpuZeroCenteredRMSNorm(nn.Module):
    def __init__(self, weight, eps):
        super().__init__()
        self.eps = eps
        self.register_buffer("scale", 1.0 + weight.float(), persistent=False)

    def forward(self, x):
        return rmsnorm(x.float().contiguous(), self.scale, self.eps).to(x.dtype)


class NpuGatedDeltaNet(nn.Module):
    def __init__(
        self,
        *,
        layer_idx,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_kernel_size,
        activation,
        eps,
        prefill_algorithm,
        chunk_size,
    ):
        super().__init__()
        if prefill_algorithm not in ("chunk", "recurrent"):
            raise ValueError(f"Unknown algorithm: {prefill_algorithm}")
        self.layer_idx = layer_idx
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_kernel_size = conv_kernel_size
        self.activation = activation
        self.prefill_algorithm = prefill_algorithm
        self.chunk_size = chunk_size
        hidden_in = None
        self.packed_proj = None
        self.conv_weight = nn.Parameter(torch.empty(0))
        self.A_log = nn.Parameter(torch.empty(0))
        self.dt_bias = nn.Parameter(torch.empty(0))
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=eps)
        self.out_proj = nn.Linear(self.value_dim, 1, bias=False)
        self._hidden_placeholder = hidden_in

    def load_local(self, checkpoint, prefix, hidden_size, dtype, rank, tp_size):
        key_groups = [self.num_k_heads * self.head_k_dim * tp_size] * 2
        value_group = self.num_v_heads * self.head_v_dim * tp_size
        qkv = checkpoint.grouped(
            f"{prefix}.in_proj_qkv.weight",
            [key_groups[0], key_groups[1], value_group],
            rank,
            tp_size,
            dim=0,
            dtype=dtype,
        )
        z = checkpoint.grouped(
            f"{prefix}.in_proj_z.weight",
            [value_group],
            rank,
            tp_size,
            dim=0,
            dtype=dtype,
        )
        heads = self.num_v_heads * tp_size
        b = checkpoint.grouped(
            f"{prefix}.in_proj_b.weight", [heads], rank, tp_size, dim=0, dtype=dtype
        )
        a = checkpoint.grouped(
            f"{prefix}.in_proj_a.weight", [heads], rank, tp_size, dim=0, dtype=dtype
        )
        packed = torch.cat((qkv, z, b, a), dim=0)
        self.packed_proj = nn.Linear(hidden_size, packed.shape[0], bias=False)
        self.packed_proj.weight = nn.Parameter(packed, requires_grad=False)
        conv = checkpoint.grouped(
            f"{prefix}.conv1d.weight",
            [key_groups[0], key_groups[1], value_group],
            rank,
            tp_size,
            dim=0,
            dtype=dtype,
        )
        self.conv_weight = nn.Parameter(conv, requires_grad=False)
        self.A_log = nn.Parameter(
            checkpoint.grouped(
                f"{prefix}.A_log", [heads], rank, tp_size, dim=0, dtype=torch.float32
            ),
            requires_grad=False,
        )
        self.dt_bias = nn.Parameter(
            checkpoint.grouped(
                f"{prefix}.dt_bias", [heads], rank, tp_size, dim=0, dtype=torch.float32
            ),
            requires_grad=False,
        )
        self.norm.weight = nn.Parameter(
            checkpoint.full(f"{prefix}.norm.weight", dtype), requires_grad=False
        )
        out = checkpoint.grouped(
            f"{prefix}.out_proj.weight",
            [value_group],
            rank,
            tp_size,
            dim=1,
            dtype=dtype,
        )
        self.out_proj = nn.Linear(out.shape[1], out.shape[0], bias=False)
        self.out_proj.weight = nn.Parameter(out, requires_grad=False)

    def core(self, hidden_states, cache_params, *, use_cache):
        batch, length, _ = hidden_states.shape
        mixed, z, b, a = self.packed_proj(hidden_states).split(
            [
                2 * self.key_dim + self.value_dim,
                self.value_dim,
                self.num_v_heads,
                self.num_v_heads,
            ],
            dim=-1,
        )
        previous = cache_params is not None and cache_params.has_previous_state(
            self.layer_idx, state_idx=0
        )
        conv_state = (
            cache_params.layers[self.layer_idx].conv_states[0] if previous else None
        )
        conv_weight = self.conv_weight
        if conv_weight.ndim == 3:
            conv_weight = conv_weight.squeeze(1)
        mixed, updated_conv = causal_conv(
            mixed.transpose(1, 2),
            conv_weight,
            None,
            state=conv_state,
            activation=self.activation,
        )
        if cache_params is not None and use_cache:
            cache_params.update_conv_state(
                updated_conv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
            )
        q, k, v = mixed.transpose(1, 2).split(
            [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        q = q.reshape(batch, length, self.num_k_heads, self.head_k_dim)
        k = k.reshape_as(q)
        v = v.reshape(batch, length, self.num_v_heads, self.head_v_dim)
        if self.num_v_heads != self.num_k_heads:
            repeat = self.num_v_heads // self.num_k_heads
            q, k = [x.repeat_interleave(repeat, dim=2) for x in (q, k)]
        decay = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        beta = b.sigmoid()
        state = (
            cache_params.layers[self.layer_idx]
            .recurrent_states[0]
            .transpose(-1, -2)
            .contiguous()
            if previous
            else None
        )
        if length == 1 or self.prefill_algorithm == "recurrent":
            output, state = gdn_recurrent(
                q, k, v, decay, beta, initial_state=state, qk_l2norm=True
            )
        else:
            output, state = gdn_chunk(
                q,
                k,
                v,
                decay,
                beta,
                initial_state=state,
                qk_l2norm=True,
                chunk_size=self.chunk_size,
            )
        if cache_params is not None and use_cache:
            cache_params.update_recurrent_state(
                state.transpose(-1, -2).contiguous(), self.layer_idx
            )
        output = self.norm(
            output.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
        )
        return output.reshape(batch, length, self.value_dim)

    def forward(self, hidden_states, cache_params=None, attention_mask=None, *, use_cache):
        if attention_mask is not None:
            raise NotImplementedError("This experiment runner requires unpadded requests")
        features = self.core(hidden_states, cache_params, use_cache=use_cache)
        return allreduce_sum(self.out_proj(features))


class NpuAttention(nn.Module):
    def __init__(self, layer_idx, head_dim, num_heads, num_kv_heads):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.scaling = head_dim**-0.5
        self.q_proj = nn.Linear(1, num_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(1, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(1, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, 1, bias=False)
        self.q_norm = None
        self.k_norm = None

    def load_local(self, checkpoint, prefix, hidden_size, eps, dtype, rank, tp_size):
        # One contiguous block per projection. Half of that block is whole heads:
        # q keeps q and gate together (2 * head_dim), and K/V/O are not split mid-head.
        q_groups = [self.head_dim * 2 * self.num_heads * tp_size]
        kv_groups = [self.head_dim * self.num_kv_heads * tp_size]
        o_groups = [self.head_dim * self.num_heads * tp_size]
        q = checkpoint.grouped(
            f"{prefix}.q_proj.weight", q_groups, rank, tp_size, dim=0, dtype=dtype
        )
        k = checkpoint.grouped(
            f"{prefix}.k_proj.weight", kv_groups, rank, tp_size, dim=0, dtype=dtype
        )
        v = checkpoint.grouped(
            f"{prefix}.v_proj.weight", kv_groups, rank, tp_size, dim=0, dtype=dtype
        )
        o = checkpoint.grouped(
            f"{prefix}.o_proj.weight", o_groups, rank, tp_size, dim=1, dtype=dtype
        )
        self.q_proj = nn.Linear(hidden_size, q.shape[0], bias=False)
        self.k_proj = nn.Linear(hidden_size, k.shape[0], bias=False)
        self.v_proj = nn.Linear(hidden_size, v.shape[0], bias=False)
        self.o_proj = nn.Linear(o.shape[1], o.shape[0], bias=False)
        self.q_proj.weight = nn.Parameter(q, requires_grad=False)
        self.k_proj.weight = nn.Parameter(k, requires_grad=False)
        self.v_proj.weight = nn.Parameter(v, requires_grad=False)
        self.o_proj.weight = nn.Parameter(o, requires_grad=False)
        self.q_norm = NpuZeroCenteredRMSNorm(
            checkpoint.full(f"{prefix}.q_norm.weight", dtype), eps
        )
        self.k_norm = NpuZeroCenteredRMSNorm(
            checkpoint.full(f"{prefix}.k_norm.weight", dtype), eps
        )

    def core(self, hidden_states, position_embeddings, past_key_values, *, use_cache):
        batch, length, _ = hidden_states.shape
        q, gate = (
            self.q_proj(hidden_states)
            .view(batch, length, self.num_heads, self.head_dim * 2)
            .chunk(2, dim=-1)
        )
        q = self.q_norm(q.contiguous()).transpose(1, 2)
        k = self.k_norm(
            self.k_proj(hidden_states).view(batch, length, self.num_kv_heads, self.head_dim)
        ).transpose(1, 2)
        v = (
            self.v_proj(hidden_states)
            .view(batch, length, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        if past_key_values is not None and use_cache:
            k, v = past_key_values.update(k, v, self.layer_idx)
        kv_length = k.shape[2]
        mask = None
        if kv_length == length:
            mask = torch.ones(length, length, dtype=torch.bool, device=q.device).triu(1)
        elif length > 1:
            mask = (
                torch.arange(kv_length, device=q.device)[None, :]
                > torch.arange(kv_length - length, kv_length, device=q.device)[:, None]
            )
        output = mha_cached(q, k, v, mask=mask, scale=self.scaling).transpose(1, 2)
        return (
            output.reshape(batch, length, -1) * gate.reshape(batch, length, -1).sigmoid()
        )

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        position_ids=None,
        *,
        use_cache,
    ):
        del position_ids
        if attention_mask is not None:
            raise NotImplementedError("This runner accepts unpadded requests")
        features = self.core(
            hidden_states, position_embeddings, past_key_values, use_cache=use_cache
        )
        return allreduce_sum(self.o_proj(features)), None


class NpuMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(1, 1, bias=False)
        self.up_proj = nn.Linear(1, 1, bias=False)
        self.down_proj = nn.Linear(1, 1, bias=False)

    def load_local(self, checkpoint, prefix, hidden_size, intermediate, dtype, rank, tp_size):
        groups = [intermediate * tp_size]
        gate = checkpoint.grouped(
            f"{prefix}.gate_proj.weight", groups, rank, tp_size, dim=0, dtype=dtype
        )
        up = checkpoint.grouped(
            f"{prefix}.up_proj.weight", groups, rank, tp_size, dim=0, dtype=dtype
        )
        down = checkpoint.grouped(
            f"{prefix}.down_proj.weight", groups, rank, tp_size, dim=1, dtype=dtype
        )
        self.gate_proj = nn.Linear(hidden_size, gate.shape[0], bias=False)
        self.up_proj = nn.Linear(hidden_size, up.shape[0], bias=False)
        self.down_proj = nn.Linear(down.shape[1], down.shape[0], bias=False)
        self.gate_proj.weight = nn.Parameter(gate, requires_grad=False)
        self.up_proj.weight = nn.Parameter(up, requires_grad=False)
        self.down_proj.weight = nn.Parameter(down, requires_grad=False)

    def core(self, hidden_states):
        return F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)

    def forward(self, hidden_states):
        return allreduce_sum(self.down_proj(self.core(hidden_states)))


class NpuDecoderLayer(nn.Module):
    def __init__(self, block_type, mixer, mlp, input_norm, post_norm):
        super().__init__()
        self.block_type = block_type
        self.mixer = mixer
        self.mlp = mlp
        self.input_layernorm = input_norm
        self.post_attention_layernorm = post_norm
        if block_type == "linear_attention":
            self.linear_attn = mixer
        else:
            self.self_attn = mixer

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kwargs,
    ):
        use_cache = kwargs.get("use_cache", True)
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            core = self.mixer.core(normed, past_key_values, use_cache=use_cache)
            proj = self.mixer.out_proj
        else:
            core = self.mixer.core(
                normed, position_embeddings, past_key_values, use_cache=use_cache
            )
            proj = self.mixer.o_proj

        split = None
        if current().overlap and current().tp_size > 1:
            from tokenspeed.runtime.models.qwen3_5_npu_parallel import choose_split

            split = choose_split(core)

        def during_second(reduced_first):
            dim, mid = split
            rest = residual.narrow(dim, 0, mid)
            return self.mlp.core(self.post_attention_layernorm(rest + reduced_first))

        overlap = split is not None
        reduced, mlp_first = project_and_reduce(
            core, proj, during_second if overlap else None
        )
        if mlp_first is None:
            hidden_states = residual + reduced
            hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        else:
            dim, mid = split
            first_hidden = residual.narrow(dim, 0, mid) + reduced.narrow(dim, 0, mid)
            second_hidden = residual.narrow(dim, mid, residual.shape[dim] - mid) + reduced.narrow(
                dim, mid, reduced.shape[dim] - mid
            )
            mlp_second = self.mlp.core(self.post_attention_layernorm(second_hidden))
            mlp_core = torch.cat((mlp_first, mlp_second), dim=dim)
            mlp_residual = torch.cat((first_hidden, second_hidden), dim=dim)
            mlp_out, _ = project_and_reduce(mlp_core, self.mlp.down_proj)
            hidden_states = mlp_residual + mlp_out
        cfg = current()
        if cfg.capture_hidden:
            cfg.hidden_layers.append(hidden_states[:, -1, :].detach().float().cpu())
        return hidden_states


class Checkpoint:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        index = model_dir / "model.safetensors.index.json"
        self.weight_map = None
        if index.exists():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        self._handles = {}
        self._prefix = None

    def _handle(self, key):
        filename = self.weight_map[key] if self.weight_map else self._single_file(key)
        handle = self._handles.get(filename)
        if handle is None:
            handle = safe_open(self.model_dir / filename, framework="pt", device="cpu")
            self._handles[filename] = handle
        return handle

    def _single_file(self, key):
        for path in sorted(self.model_dir.glob("*.safetensors")):
            handle = self._handles.get(path.name)
            if handle is None:
                handle = safe_open(path, framework="pt", device="cpu")
                self._handles[path.name] = handle
            if key in handle.keys():
                return path.name
        raise KeyError(key)

    def resolve(self, suffix: str) -> str:
        candidates = [
            "model.language_model." + suffix[len("model.") :],
            suffix,
            "language_model." + suffix[len("model.") :],
        ]
        if self.weight_map:
            for key in candidates:
                if key in self.weight_map:
                    return key
        else:
            for key in candidates:
                try:
                    self._handle(key)
                    if key in self._handles[self._single_file(key)].keys():
                        return key
                except KeyError:
                    continue
        raise KeyError(suffix)

    def _raw(self, suffix):
        key = self.resolve(suffix)
        return self._handle(key).get_tensor(key)

    def full(self, suffix, dtype):
        return self._raw(suffix).to(dtype)

    def full_exact(self, key, dtype):
        return self._handle(key).get_tensor(key).to(dtype)

    def grouped(self, suffix, groups, rank, tp_size, *, dim, dtype):
        return grouped_slice(self._raw(suffix), groups, rank, tp_size, dim).to(dtype).contiguous()

    def close(self):
        self._handles.clear()


def _record(name, tensor):
    current().shard_shapes[name] = list(tensor.shape)


def load_text_model(
    model_dir,
    *,
    device,
    dtype,
    backend,
    prefill_algorithm,
    chunk_size,
    fuse_projections,
    tp_rank=0,
    tp_size=1,
    overlap=False,
):
    """Load text weights. TP shards each projection; embedding stays whole."""
    del fuse_projections
    if backend not in ("reference", "ascend"):
        raise ValueError(f"Unknown backend: {backend}")
    path = Path(model_dir)
    raw = json.loads((path / "config.json").read_text())
    text = raw["text_config"] if "text_config" in raw else raw
    config = Qwen3_5TextConfig(**text)
    tie_embeddings = bool(text.get("tie_word_embeddings", raw.get("tie_word_embeddings", True)))
    config._attn_implementation = "eager"
    require_divisible(config, tp_size)
    install(
        TPConfig(
            rank=tp_rank,
            tp_size=tp_size,
            group=tuple(range(tp_size)),
            overlap=overlap,
        )
    )
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    if backend != "ascend":
        raise ValueError("reference loading is unchanged only for the eager NPU path")
    checkpoint = Checkpoint(path)
    hidden = config.hidden_size
    eps = config.rms_norm_eps
    local_k = config.linear_num_key_heads // tp_size
    local_v = config.linear_num_value_heads // tp_size
    local_q = config.num_attention_heads // tp_size
    local_kv = config.num_key_value_heads // tp_size
    local_i = config.intermediate_size // tp_size
    layers = []
    for index, block_type in enumerate(config.layer_types):
        prefix = f"model.layers.{index}"
        if block_type == "linear_attention":
            mixer = NpuGatedDeltaNet(
                layer_idx=index,
                num_k_heads=local_k,
                num_v_heads=local_v,
                head_k_dim=config.linear_key_head_dim,
                head_v_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                activation=config.hidden_act,
                eps=eps,
                prefill_algorithm=prefill_algorithm,
                chunk_size=chunk_size,
            )
            mixer.load_local(
                checkpoint,
                f"{prefix}.linear_attn",
                hidden,
                dtype,
                tp_rank,
                tp_size,
            )
            _record(f"layer{index}.gdn_out", mixer.out_proj.weight)
            _record(f"layer{index}.gdn_conv", mixer.conv_weight)
        else:
            mixer = NpuAttention(index, config.head_dim, local_q, local_kv)
            mixer.load_local(
                checkpoint, f"{prefix}.self_attn", hidden, eps, dtype, tp_rank, tp_size
            )
            _record(f"layer{index}.q_proj", mixer.q_proj.weight)
            _record(f"layer{index}.k_proj", mixer.k_proj.weight)
        mlp = NpuMLP()
        mlp.load_local(
            checkpoint, f"{prefix}.mlp", hidden, local_i, dtype, tp_rank, tp_size
        )
        if index == 0:
            _record("mlp.down", mlp.down_proj.weight)
            _record("mlp.gate", mlp.gate_proj.weight)
        input_norm = NpuZeroCenteredRMSNorm(
            checkpoint.full(f"{prefix}.input_layernorm.weight", dtype), eps
        )
        post_norm = NpuZeroCenteredRMSNorm(
            checkpoint.full(f"{prefix}.post_attention_layernorm.weight", dtype), eps
        )
        layers.append(NpuDecoderLayer(block_type, mixer, mlp, input_norm, post_norm))
    embed = checkpoint.full("model.embed_tokens.weight", dtype)
    if tie_embeddings:
        lm_weight = embed
    else:
        lm_weight = checkpoint.full_exact("lm_head.weight", dtype)
    final = NpuZeroCenteredRMSNorm(checkpoint.full("model.norm.weight", dtype), eps)
    checkpoint.close()
    model.model.layers = nn.ModuleList(layers)
    model.model.embed_tokens = nn.Embedding.from_pretrained(embed, freeze=True)
    model.model.norm = final
    model.lm_head = nn.Linear(hidden, lm_weight.shape[0], bias=False)
    if tie_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    else:
        model.lm_head.weight = nn.Parameter(lm_weight, requires_grad=False)
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
    model.requires_grad_(False)
    model.tp_config = current()
    _record("embed", model.model.embed_tokens.weight)
    missing = [
        name
        for name, param in model.named_parameters()
        if getattr(param, "is_meta", False) or param.device.type == "meta"
    ]
    if missing:
        raise RuntimeError(f"meta parameters remain: {missing[:8]}")
    return model.to(device).eval()


def forward_request(model, input_ids, *, cache, backend, logits_to_keep):
    """Pass caller-owned request cache explicitly; NPU requests are unpadded."""
    if backend not in ("ascend", "reference"):
        raise ValueError(f"Unknown backend: {backend}")
    mask = (
        {"full_attention": None, "linear_attention": None}
        if backend == "ascend"
        else None
    )
    return model(
        input_ids=input_ids,
        attention_mask=mask,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=logits_to_keep,
    )
