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

"""One eager TP worker. ASCEND_RT_VISIBLE_DEVICES is already a single chip."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

from tokenspeed.runtime.models.qwen3_5_npu import forward_request, load_text_model
from tokenspeed.runtime.models.qwen3_5_npu_parallel import (
    begin_region,
    current,
    drain_timeline,
    init_hccl,
)


def _sync() -> None:
    torch.npu.synchronize()


def _tensors(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        found = []
        for item in value.values():
            found.extend(_tensors(item))
        return found
    if isinstance(value, (list, tuple)):
        found = []
        for item in value:
            found.extend(_tensors(item))
        return found
    return []


def _cache_bytes(cache) -> dict[str, int]:
    kv = conv = recurrent = 0
    for layer in getattr(cache, "layers", []) or []:
        for key in _tensors(getattr(layer, "keys", None)):
            kv += key.numel() * key.element_size()
        for value in _tensors(getattr(layer, "values", None)):
            kv += value.numel() * value.element_size()
        for state in _tensors(getattr(layer, "conv_states", None)):
            conv += state.numel() * state.element_size()
        for state in _tensors(getattr(layer, "recurrent_states", None)):
            recurrent += state.numel() * state.element_size()
    return {"kv_bytes": kv, "conv_bytes": conv, "gdn_state_bytes": recurrent}


def _agree(token: torch.Tensor, world: int) -> bool:
    if world == 1:
        return True
    gathered = [torch.empty_like(token) for _ in range(world)]
    dist.all_gather(gathered, token)
    return all(torch.equal(gathered[0], item) for item in gathered)


def _generate(model, input_ids, steps: int):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    cache = DynamicCache(config=model.config)
    tokens = []
    _sync()
    current().pending.clear()
    begin_region()
    started = time.perf_counter()
    out = forward_request(model, input_ids, cache=cache, backend="ascend", logits_to_keep=1)
    _sync()
    prefill_s = time.perf_counter() - started
    prefill_timeline = drain_timeline()
    next_token = out.logits[:, -1].argmax(dim=-1)
    if os.environ.get("HW2_SKIP_ALLREDUCE") != "1" and not _agree(next_token, world):
        raise RuntimeError("ranks disagree on the first token")
    tokens.append(int(next_token[0].item()) if next_token.ndim == 1 else int(next_token.view(-1)[0]))
    decode_s = 0.0
    for _ in range(steps - 1):
        step_ids = next_token.view(-1, 1)
        _sync()
        started = time.perf_counter()
        out = forward_request(model, step_ids, cache=cache, backend="ascend", logits_to_keep=1)
        _sync()
        decode_s += time.perf_counter() - started
        next_token = out.logits[:, -1].argmax(dim=-1)
        if os.environ.get("HW2_SKIP_ALLREDUCE") != "1" and not _agree(next_token, world):
            raise RuntimeError("ranks disagree during decode")
        tokens.append(int(next_token.view(-1)[0]))
    return {
        "tokens": tokens,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "decode_steps": max(steps - 1, 0),
        "cache": _cache_bytes(cache),
        "timeline": prefill_timeline,
        "decode_timeline": drain_timeline(),
    }


def _stats(model) -> dict:
    _sync()
    return {
        "rank": int(os.environ.get("RANK", "0")),
        "physical_chip": os.environ.get("HW2_PHYSICAL_CHIP"),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "logical_device": str(torch.npu.current_device()),
        "device_count": torch.npu.device_count(),
        "allocated_bytes": torch.npu.max_memory_allocated(),
        "reserved_bytes": torch.npu.max_memory_reserved(),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "shard_shapes": current().shard_shapes,
        "comm_count": current().comm_count,
        "comm_payload_bytes": current().comm_payload_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--mode", choices=("smoke", "perf", "logits", "isolate", "layers"), default="smoke")
    parser.add_argument("--prompt", default="请用一句话介绍北京大学。")
    parser.add_argument("--seq-len", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.npu.set_device(0)
    if world > 1:
        init_hccl(rank, world, f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    torch.npu.reset_peak_memory_stats()
    model = load_text_model(
        args.model_dir,
        device="npu:0",
        dtype=torch.bfloat16,
        backend="ascend",
        prefill_algorithm="chunk",
        chunk_size=64,
        fuse_projections=True,
        tp_rank=rank,
        tp_size=world,
        overlap=args.overlap,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if args.seq_len:
        piece = tokenizer(args.prompt, add_special_tokens=False).input_ids or [1]
        ids = (piece * (args.seq_len // len(piece) + 1))[: args.seq_len]
    else:
        ids = tokenizer(args.prompt, add_special_tokens=True).input_ids
    if args.mode == "isolate":
        _isolate(model, tokenizer, Path(args.output))
        return
    if args.mode == "layers":
        _save_layers(model, ids, args, rank, world)
        return
    batch = torch.tensor([ids] * args.batch, dtype=torch.long, device="npu:0")
    for _ in range(args.warmup):
        _generate(model, batch, 2)
        current().pending.clear()
        current().comm_count = 0
        current().comm_payload_bytes = 0
    torch.npu.reset_peak_memory_stats()
    result = _generate(model, batch, args.steps)
    cache_bytes = result.pop("cache")
    record = {
        "mode": args.mode,
        "model_dir": args.model_dir,
        "seq_len": batch.shape[1],
        "batch": batch.shape[0],
        "steps": args.steps,
        "overlap": args.overlap,
        "prompt_tokens": result["tokens"],
        **{key: value for key, value in result.items() if key != "tokens"},
        "tokens": result["tokens"],
        **_stats(model),
        **cache_bytes,
    }
    decode_steps = result["decode_steps"]
    record["prefill_ms"] = result["prefill_s"] * 1e3
    record["tpot_ms"] = (result["decode_s"] / decode_steps * 1e3) if decode_steps else None
    total_s = result["prefill_s"] + result["decode_s"]
    record["e2e_tokens_per_s"] = (batch.shape[0] * (batch.shape[1] + args.steps)) / total_s
    # Per sequence. One decode step emits one token on every row, so system
    # throughput is this value times the batch.
    per_sequence = (1e3 / record["tpot_ms"]) if record["tpot_ms"] else None
    record["decode_tokens_per_s"] = per_sequence
    record["aggregate_decode_tokens_per_s"] = (
        None if per_sequence is None else batch.shape[0] * per_sequence
    )
    record["prefill_tokens_per_s"] = (batch.shape[0] * batch.shape[1]) / result["prefill_s"]
    text = tokenizer.decode(result["tokens"], skip_special_tokens=True)
    record["text"] = text
    if args.mode == "logits":
        logits = _last_logits(model, batch)
        record["logits_checksum"] = float(logits.float().sum())
        if rank == 0:
            torch.save(logits.float().cpu(), Path(args.output).with_suffix(".pt"))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if world == 1 or rank == 0:
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    else:
        path.with_name(path.stem + f".rank{rank}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2)
        )
    print(json.dumps({k: record[k] for k in ("rank", "physical_chip", "prefill_ms", "tpot_ms", "text", "shard_shapes") if k in record}, ensure_ascii=False)[:2000])


def _save_layers(model, ids, args, rank: int, world: int) -> None:
    """One prefill. Each layer's last-token hidden state is kept for TP=1 vs TP=2."""
    cfg = current()
    cfg.capture_hidden = True
    cfg.hidden_layers = []
    batch = torch.tensor([ids] * args.batch, dtype=torch.long, device="npu:0")
    cache = DynamicCache(config=model.config)
    out = forward_request(model, batch, cache=cache, backend="ascend", logits_to_keep=1)
    hidden = torch.stack(cfg.hidden_layers)
    payload = {
        "hidden_last": hidden,
        "logits": out.logits[:, -1].float().cpu(),
        "seq_len": batch.shape[1],
        "layers": hidden.shape[0],
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path if world == 1 or rank == 0 else path.with_name(path.stem + f".rank{rank}.pt")
    if target.suffix != ".pt":
        target = target.with_suffix(".pt")
    torch.save(payload, target)
    record = {
        "mode": "layers",
        "rank": rank,
        "physical_chip": os.environ.get("HW2_PHYSICAL_CHIP"),
        "seq_len": batch.shape[1],
        "layers": int(hidden.shape[0]),
        "hidden_shape": list(hidden.shape),
    }
    text_path = path if world == 1 or rank == 0 else path.with_name(path.stem + f".rank{rank}.json")
    text_path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


def _isolate(model, tokenizer, output: Path) -> None:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    left = tokenizer("请用一句话介绍北京大学。", return_tensors="pt").input_ids[0].tolist()
    right = tokenizer("计算 12 乘以 13，只给出结果。", return_tensors="pt").input_ids[0].tolist()
    width = max(len(left), len(right))
    # Equal length, no padding token: repeat each prompt's own tokens.
    left = (left * (width // len(left) + 1))[:width]
    right = (right * (width // len(right) + 1))[:width]
    device = "npu:0"

    def once(rows):
        batch = torch.tensor(rows, dtype=torch.long, device=device)
        cache = DynamicCache(config=model.config)
        out = forward_request(model, batch, cache=cache, backend="ascend", logits_to_keep=1)
        return out.logits[:, -1].float(), cache

    both, cache = once([left, right])
    only_left, _ = once([left])
    only_right, _ = once([right])
    def nrmse(a, b):
        d = (a - b).float()
        return float(d.pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt())
    states = []
    for layer in cache.layers:
        for state in _tensors(getattr(layer, "recurrent_states", None)):
            states.append(state)
    record = {
        "rank": int(os.environ.get("RANK", "0")),
        "physical_chip": os.environ.get("HW2_PHYSICAL_CHIP"),
        "left_nrmse": nrmse(both[0], only_left[0]),
        "right_nrmse": nrmse(both[1], only_right[0]),
        "left_top1": int(both[0].argmax()) == int(only_left[0].argmax()),
        "right_top1": int(both[1].argmax()) == int(only_right[0].argmax()),
        "state_rows": int(states[0].shape[0]) if states else 0,
        "state_independent": bool(states) and states[0].shape[0] == 2 and not torch.equal(states[0][0], states[0][1]),
        "ranks_agree": _agree(both.argmax(dim=-1), world),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    target = output if world == 1 or record["rank"] == 0 else output.with_name(output.stem + f".rank{record['rank']}.json")
    target.write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


def _last_logits(model, input_ids):
    cache = DynamicCache(config=model.config)
    out = forward_request(model, input_ids, cache=cache, backend="ascend", logits_to_keep=1)
    return out.logits[:, -1]


if __name__ == "__main__":
    main()
