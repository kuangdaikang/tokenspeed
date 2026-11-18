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

"""Ascend eager GDN scans for the shared paged-state backend."""

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.gdn import GdnChunkPrefillResult
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel_npu.ops.gdn import gdn_chunk, gdn_recurrent

_CAPABILITY = CapabilityRequirement(vendors=frozenset({"ascend"}))
_SIGNATURES = format_signatures(
    ("q", "k", "v"), "dense", {torch.bfloat16, torch.float16}
)


def _expand_query_key(q, k, v):
    num_groups, remainder = divmod(v.shape[-2], q.shape[-2])
    if not num_groups or remainder:
        raise ValueError(
            "GDN value head count must be a multiple of the query head count"
        )
    if num_groups > 1:
        q = q.repeat_interleave(num_groups, dim=-2)
        k = k.repeat_interleave(num_groups, dim=-2)
    return q, k


@register_kernel(
    "attention",
    "gdn_chunk_prefill",
    name="ascend_gdn_chunk_prefill",
    solution="ascend",
    capability=_CAPABILITY,
    signatures=_SIGNATURES,
    traits={"output_h": frozenset({False})},
    priority=Priority.PORTABLE,
)
def ascend_gdn_chunk_prefill(
    q,
    k,
    v,
    g,
    beta,
    *,
    scale,
    initial_state,
    cu_seqlens,
    qk_l2norm,
    output_final_state,
    output_h,
):
    """Scan packed requests and return their K-last states in batch order."""
    if output_h:
        raise ValueError("Ascend GDN prefill only returns final checkpoints")
    q, k = _expand_query_key(q, k, v)
    boundaries = cu_seqlens.tolist()
    output = torch.zeros_like(v, dtype=q.dtype)
    final_state = initial_state.float().clone() if output_final_state else None
    for row, (begin, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        if begin == end:
            continue
        out, state = gdn_chunk(
            q[:, begin:end],
            k[:, begin:end],
            v[:, begin:end],
            g[:, begin:end],
            beta[:, begin:end],
            initial_state=initial_state[row : row + 1],
            qk_l2norm=qk_l2norm,
            chunk_size=64,
            scale=scale,
        )
        output[:, begin:end].copy_(out)
        if final_state is not None:
            final_state[row : row + 1].copy_(state)
    return GdnChunkPrefillResult(output, final_state)


@register_kernel(
    "attention",
    "gdn_decode_step",
    name="ascend_gdn_decode_step",
    solution="ascend",
    capability=_CAPABILITY,
    signatures=_SIGNATURES,
    priority=Priority.PORTABLE,
)
def ascend_gdn_decode_step(
    q,
    k,
    v,
    *,
    A_log,
    a,
    dt_bias,
    b,
    initial_state,
    initial_state_indices,
    scale,
    output_state_indices,
    use_qk_l2norm,
):
    """Evolve each live request's state into the scheduler's output block."""
    q, k = _expand_query_key(q, k, v)
    reads = initial_state_indices.long()
    writes = reads if output_state_indices is None else output_state_indices.long()
    live = (reads >= 0) & (writes >= 0)
    state = initial_state.index_select(0, reads.clamp_min(0))
    state.masked_fill_(~live[:, None, None, None], 0)
    decay = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    out, state = gdn_recurrent(
        q,
        k,
        v,
        decay,
        b.float().sigmoid(),
        initial_state=state,
        qk_l2norm=use_qk_l2norm,
        scale=scale,
    )
    initial_state.index_copy_(0, writes[live], state[live].to(initial_state.dtype))
    return out.masked_fill_(~live[:, None, None, None], 0)
