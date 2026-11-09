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

"""TP helpers for the Qwen3.5 Ascend eager runner.

Collectives go through the existing HCCL backend. A process group is usable
only after ``process_group_manager`` has registered it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch


@dataclass
class TPConfig:
    rank: int = 0
    tp_size: int = 1
    group: tuple[int, ...] = (0,)
    overlap: bool = False
    comm_stream: object | None = None
    pending: list = field(default_factory=list)
    comm_payload_bytes: int = 0
    comm_count: int = 0
    shard_shapes: dict = field(default_factory=dict)
    base_event: object | None = None
    capture_hidden: bool = False
    hidden_layers: list = field(default_factory=list)


_CURRENT: TPConfig | None = None


def current() -> TPConfig:
    if _CURRENT is None:
        return TPConfig()
    return _CURRENT


def install(config: TPConfig) -> TPConfig:
    global _CURRENT
    _CURRENT = config
    return config


def require_divisible(config, tp_size: int) -> None:
    """Refuse a silent head or intermediate replication."""
    fields = {
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "linear_num_key_heads": config.linear_num_key_heads,
        "linear_num_value_heads": config.linear_num_value_heads,
        "intermediate_size": config.intermediate_size,
    }
    bad = {name: value for name, value in fields.items() if value % tp_size}
    if bad:
        raise RuntimeError(
            f"TP={tp_size} does not divide {bad}. "
            "Refusing to replicate those dimensions."
        )


def init_hccl(rank: int, world_size: int, init_method: str) -> tuple[int, ...]:
    """Initialize HCCL the same way DistributedInitializer does for NPU.

    ``device_id`` stays None. Every rank then registers the same rank tuple
    before any collective.
    """
    from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend
    from tokenspeed.runtime.distributed.comm_backend.registry import (
        set_global_backend,
    )
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )

    torch.npu.set_device(0)
    pg_manager.init_distributed(
        SimpleNamespace(world_size=world_size, rank=rank),
        distributed_init_method=init_method,
        backend="hccl",
        timeout=1800,
        device_id=None,
    )
    group = tuple(range(world_size))
    pg_manager.init_process_group(group, backend="hccl")
    set_global_backend(HcclBackend())
    return group


def _backend():
    from tokenspeed.runtime.distributed.comm_backend.registry import get_global_backend

    return get_global_backend()


def allreduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """In-place SUM AllReduce on the current stream. TP=1 is a no-op."""
    cfg = current()
    if cfg.tp_size == 1 or os.environ.get("HW2_SKIP_ALLREDUCE") == "1":
        return tensor
    tensor = tensor.contiguous()
    start, end = _events()
    if start is not None:
        start.record()
    _backend().all_reduce(tensor, cfg.group)
    if end is not None:
        end.record()
        cfg.pending.append(("allreduce", "comm", start, end))
    cfg.comm_count += 1
    cfg.comm_payload_bytes += tensor.numel() * tensor.element_size()
    return tensor


def _events():
    if not torch.npu.is_available():
        return None, None
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    return start, end


def _comm_stream():
    cfg = current()
    if cfg.comm_stream is None:
        cfg.comm_stream = torch.npu.Stream()
    return cfg.comm_stream


def launch_allreduce(tensor: torch.Tensor):
    """Queue SUM AllReduce on the communication stream. Returns its end event."""
    cfg = current()
    tensor = tensor.contiguous()
    if cfg.tp_size == 1 or os.environ.get("HW2_SKIP_ALLREDUCE") == "1":
        return None
    compute = torch.npu.current_stream()
    comm = _comm_stream()
    comm.wait_stream(compute)
    start, end = _events()
    with torch.npu.stream(comm):
        start.record()
        _backend().all_reduce(tensor, cfg.group)
        end.record()
    cfg.pending.append(("allreduce", "comm", start, end))
    cfg.comm_count += 1
    cfg.comm_payload_bytes += tensor.numel() * tensor.element_size()
    return end


def wait_event(event) -> None:
    if event is None:
        return
    torch.npu.current_stream().wait_event(event)


def choose_split(tensor: torch.Tensor) -> tuple[int, int] | None:
    """Split tokens when the sequence allows it, otherwise the batch."""
    if tensor.shape[1] >= 2:
        return 1, tensor.shape[1] // 2
    if tensor.shape[0] >= 2:
        return 0, tensor.shape[0] // 2
    return None


def project_and_reduce(core: torch.Tensor, proj, during_second=None):
    """Row-parallel projection followed by SUM AllReduce.

    With overlap, AllReduce of the first chunk runs while the second chunk's
    projection runs. ``during_second`` then runs while the second AllReduce is
    in flight, which is where the next RMSNorm and projection overlap comm.
    """
    cfg = current()
    split = choose_split(core) if cfg.overlap and cfg.tp_size > 1 else None
    if split is None:
        projected = proj(core)
        return allreduce_sum(projected), None
    dim, mid = split
    first = core.narrow(dim, 0, mid)
    second = core.narrow(dim, mid, core.shape[dim] - mid)
    y0 = proj(first)
    end0 = launch_allreduce(y0)
    start, end = _events()
    start.record()
    y1 = proj(second)
    end.record()
    cfg.pending.append(("row_proj_chunk1", "compute", start, end))
    end1 = launch_allreduce(y1)
    wait_start, wait_end = _events()
    wait_start.record()
    wait_event(end0)
    wait_end.record()
    cfg.pending.append(("wait_allreduce_chunk0", "wait", wait_start, wait_end))
    extra = during_second(y0) if during_second is not None else None
    wait_start, wait_end = _events()
    wait_start.record()
    wait_event(end1)
    wait_end.record()
    cfg.pending.append(("wait_allreduce_chunk1", "wait", wait_start, wait_end))
    return torch.cat((y0, y1), dim=dim), extra


def begin_region() -> None:
    """Mark the zero point used to place later spans on one timeline."""
    cfg = current()
    if not torch.npu.is_available():
        return
    base = torch.npu.Event(enable_timing=True)
    base.record()
    cfg.base_event = base


def drain_timeline() -> list[dict]:
    """Synchronize and turn recorded events into millisecond spans."""
    cfg = current()
    if not cfg.pending:
        return []
    torch.npu.synchronize()
    spans = []
    for name, kind, start, end in cfg.pending:
        span = {
            "name": name,
            "kind": kind,
            "ms": float(start.elapsed_time(end)),
        }
        if cfg.base_event is not None:
            span["start_ms"] = float(cfg.base_event.elapsed_time(start))
            span["end_ms"] = float(cfg.base_event.elapsed_time(end))
        spans.append(span)
    cfg.pending.clear()
    return spans


def grouped_slice(tensor: torch.Tensor, groups: list[int], rank: int, tp_size: int, dim: int):
    """Take each logical group's contiguous rank slice and concatenate."""
    parts = []
    offset = 0
    for group in groups:
        if group % tp_size:
            raise RuntimeError(f"group {group} is not divisible by TP {tp_size}")
        local = group // tp_size
        start = offset + rank * local
        stop = start + local
        if dim == 0:
            parts.append(tensor[start:stop])
        elif dim == 1:
            parts.append(tensor[:, start:stop])
        else:
            raise ValueError(dim)
        offset += group
    if offset != tensor.shape[dim]:
        raise RuntimeError(
            f"groups {groups} sum to {offset}, tensor dim {dim} is {tensor.shape[dim]}"
        )
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=dim)
