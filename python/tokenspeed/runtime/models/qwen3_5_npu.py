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

"""Single-chip Qwen3.5 text adapter for reproducible Ascend experiments.

Reuses Transformers' text decoder, RoPE, and explicit per-request DynamicCache;
uses TokenSpeed's NPU kernels and K-last GDN state contract. This eager runner
is separate from the serving scheduler and does not claim serving integration.
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
    mha_prefill,
    rmsnorm,
)
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
)


class NpuZeroCenteredRMSNorm(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.weight = source.weight
        self.eps = source.eps
        # Frozen inference weight: retain FP32 1+w rather than rounding it.
        self.register_buffer("scale", 1.0 + source.weight.float(), persistent=False)

    def forward(self, x):
        return rmsnorm(x.float().contiguous(), self.scale, self.eps).to(x.dtype)


class NpuGatedDeltaNet(nn.Module):
    def __init__(self, source, *, prefill_algorithm, chunk_size, fuse_projections):
        super().__init__()
        if prefill_algorithm not in ("chunk", "recurrent"):
            raise ValueError(f"Unknown algorithm: {prefill_algorithm}")
        self.prefill_algorithm = prefill_algorithm
        self.chunk_size = chunk_size
        self.fuse_projections = fuse_projections
        self.layer_idx = source.layer_idx
        self.num_v_heads = source.num_v_heads
        self.num_k_heads = source.num_k_heads
        self.head_k_dim = source.head_k_dim
        self.head_v_dim = source.head_v_dim
        self.key_dim = source.key_dim
        self.value_dim = source.value_dim
        self.conv_kernel_size = source.conv_kernel_size
        self.activation = source.activation
        self.conv1d = source.conv1d
        self.A_log = source.A_log
        self.dt_bias = source.dt_bias
        self.norm = source.norm
        self.out_proj = source.out_proj
        self.packed_proj = None
        self.in_proj_qkv = None
        self.in_proj_z = None
        self.in_proj_b = None
        self.in_proj_a = None
        if fuse_projections:
            self.packed_proj = nn.Linear(
                source.hidden_size,
                self.key_dim * 2 + self.value_dim * 2 + self.num_v_heads * 2,
                bias=False,
                device=source.in_proj_qkv.weight.device,
                dtype=source.in_proj_qkv.weight.dtype,
            )
            self.packed_proj.weight = nn.Parameter(
                torch.cat(
                    [
                        m.weight
                        for m in (
                            source.in_proj_qkv,
                            source.in_proj_z,
                            source.in_proj_b,
                            source.in_proj_a,
                        )
                    ],
                    dim=0,
                ),
                requires_grad=False,
            )
        else:
            self.in_proj_qkv = source.in_proj_qkv
            self.in_proj_z = source.in_proj_z
            self.in_proj_b = source.in_proj_b
            self.in_proj_a = source.in_proj_a

    def forward(
        self, hidden_states, cache_params=None, attention_mask=None, *, use_cache
    ):
        if attention_mask is not None:
            raise NotImplementedError(
                "This experiment runner requires unpadded requests"
            )
        batch, length, _ = hidden_states.shape
        if self.fuse_projections:
            mixed, z, b, a = self.packed_proj(hidden_states).split(
                [
                    2 * self.key_dim + self.value_dim,
                    self.value_dim,
                    self.num_v_heads,
                    self.num_v_heads,
                ],
                dim=-1,
            )
        else:
            mixed, z, b, a = [
                projection(hidden_states)
                for projection in (
                    self.in_proj_qkv,
                    self.in_proj_z,
                    self.in_proj_b,
                    self.in_proj_a,
                )
            ]
        previous = cache_params is not None and cache_params.has_previous_state(
            self.layer_idx, state_idx=0
        )
        conv_state = (
            cache_params.layers[self.layer_idx].conv_states[0] if previous else None
        )
        mixed, updated_conv = causal_conv(
            mixed.transpose(1, 2),
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            state=conv_state,
            activation=self.activation,
        )
        if cache_params is not None:
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
        # HF cache uses K,V; convert explicitly at the kernel boundary.
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
        if cache_params is not None:
            cache_params.update_recurrent_state(
                state.transpose(-1, -2).contiguous(), self.layer_idx
            )
        output = self.norm(
            output.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
        )
        return self.out_proj(output.reshape(batch, length, self.value_dim))


class NpuAttention(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.layer_idx = source.layer_idx
        self.head_dim = source.head_dim
        self.scaling = source.scaling
        self.q_proj = source.q_proj
        self.k_proj = source.k_proj
        self.v_proj = source.v_proj
        self.o_proj = source.o_proj
        self.q_norm = NpuZeroCenteredRMSNorm(source.q_norm)
        self.k_norm = NpuZeroCenteredRMSNorm(source.k_norm)

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
        if attention_mask is not None:
            raise NotImplementedError("This runner accepts one unpadded request")
        batch, length, _ = hidden_states.shape
        if batch != 1:
            raise NotImplementedError("This runner supports batch size 1")
        q, gate = (
            self.q_proj(hidden_states)
            .view(batch, length, -1, self.head_dim * 2)
            .chunk(2, dim=-1)
        )
        q = self.q_norm(q.contiguous()).transpose(1, 2)
        k = self.k_norm(
            self.k_proj(hidden_states).view(batch, length, -1, self.head_dim)
        ).transpose(1, 2)
        v = (
            self.v_proj(hidden_states)
            .view(batch, length, -1, self.head_dim)
            .transpose(1, 2)
        )
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        kv_length = k.shape[2]
        if kv_length == length:
            qt, kt, vt = [x.squeeze(0).transpose(0, 1).contiguous() for x in (q, k, v)]
            output = mha_prefill(
                qt, kt, vt, None, [0, length], length, softmax_scale=self.scaling
            ).unsqueeze(0)
        else:
            mask = None
            if length > 1:
                mask = (
                    torch.arange(kv_length, device=q.device)[None, :]
                    > torch.arange(kv_length - length, kv_length, device=q.device)[
                        :, None
                    ]
                )
            output = mha_cached(q, k, v, mask=mask, scale=self.scaling)
            output = output.transpose(1, 2)
        output = (
            output.reshape(batch, length, -1)
            * gate.reshape(batch, length, -1).sigmoid()
        )
        return self.o_proj(output), None


def load_text_model(
    model_dir,
    *,
    device,
    dtype,
    backend,
    prefill_algorithm,
    chunk_size,
    fuse_projections,
):
    """Strictly load every text parameter; ignore only vision and MTP weights."""
    if backend not in ("reference", "ascend"):
        raise ValueError(f"Unknown backend: {backend}")
    path = Path(model_dir)
    config = Qwen3_5TextConfig(
        **json.loads((path / "config.json").read_text())["text_config"]
    )
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    weights = {}
    for filename in sorted(path.glob("*.safetensors")):
        with safe_open(filename, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                if name.startswith("model.language_model."):
                    key = name.replace("model.language_model.", "model.", 1)
                    value = checkpoint.get_tensor(name)
                    weights[key] = value.to(
                        torch.float32 if key.endswith(("A_log", "dt_bias")) else dtype
                    )
    weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    model.load_state_dict(weights, strict=True, assign=True)
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
    model.tie_weights()
    model.requires_grad_(False)
    if backend == "ascend":
        for layer in model.model.layers:
            if layer.block_type == "linear_attention":
                layer.linear_attn = NpuGatedDeltaNet(
                    layer.linear_attn,
                    prefill_algorithm=prefill_algorithm,
                    chunk_size=chunk_size,
                    fuse_projections=fuse_projections,
                )
            else:
                layer.self_attn = NpuAttention(layer.self_attn)
            layer.input_layernorm = NpuZeroCenteredRMSNorm(layer.input_layernorm)
            layer.post_attention_layernorm = NpuZeroCenteredRMSNorm(
                layer.post_attention_layernorm
            )
        model.model.norm = NpuZeroCenteredRMSNorm(model.model.norm)
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
