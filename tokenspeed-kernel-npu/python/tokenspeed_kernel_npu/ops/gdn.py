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

"""Dense request-local causal convolution and GDN for Ascend eager inference.

GDN state is [B,H,V,K], matching TokenSpeed's K-last state contract.
All arithmetic stays on the input device. There is no host state transfer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_conv(x, weight, bias, *, state, activation):
    """Convolve [B,C,T] with [C,W]; return output and last W raw inputs.

    State holds W inputs (including the oldest); only W-1 contribute to the
    next output. Accumulate in FP32 and cast once, matching a BF16 Conv1d.
    """
    if activation != "silu":
        raise ValueError(f"Unsupported activation: {activation}")
    width = weight.shape[-1]
    history = x.new_zeros(x.shape[0], x.shape[1], width) if state is None else state
    joined = torch.cat((history, x), dim=-1)
    # Explicit short stencil avoids thousands of depthwise Conv1d groups.
    y = torch.zeros_like(x, dtype=torch.float32)
    for i in range(width):
        y = (
            y
            + joined[:, :, 1 + i : 1 + i + x.shape[-1]].float()
            * weight[:, i].float()[None, :, None]
        )
    if bias is not None:
        y = y + bias.float()[None, :, None]
    y = F.silu(y.to(x.dtype))
    return y, joined[:, :, -width:].contiguous()


def _prepare(query, key, value, g, beta, *, qk_l2norm):
    q, k, v, decay, b = [
        x.transpose(1, 2).float().contiguous() for x in (query, key, value, g, beta)
    ]
    if qk_l2norm:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    return q * (q.shape[-1] ** -0.5), k, v, decay, b


def gdn_recurrent(query, key, value, g, beta, *, initial_state, qk_l2norm):
    """Exact delta recurrence; inputs [B,T,H,K/V], state [B,H,V,K]."""
    q, k, v, decay, b = _prepare(query, key, value, g, beta, qk_l2norm=qk_l2norm)
    state = (
        v.new_zeros(v.shape[0], v.shape[1], v.shape[-1], k.shape[-1])
        if initial_state is None
        else initial_state.float()
    )
    outputs = []
    for t in range(q.shape[2]):
        state = state * decay[:, :, t].exp()[..., None, None]
        predicted = (state * k[:, :, t].unsqueeze(-2)).sum(-1)
        delta = (v[:, :, t] - predicted) * b[:, :, t, None]
        state = state + delta.unsqueeze(-1) * k[:, :, t].unsqueeze(-2)
        outputs.append((state * q[:, :, t].unsqueeze(-2)).sum(-1))
    return (
        torch.stack(outputs, dim=2).transpose(1, 2).contiguous().to(query.dtype),
        state.contiguous(),
    )


def _diagonal_blocks(matrix, block_size):
    blocks = matrix.shape[-1] // block_size
    return (
        matrix.reshape(*matrix.shape[:-2], blocks, block_size, blocks, block_size)
        .diagonal(dim1=-4, dim2=-2)
        .movedim(-1, -3)
        .contiguous()
    )


def _unit_lower_inverse(negative_lower):
    """Invert I-L for strictly lower L, using stable block substitution."""
    length = negative_lower.shape[-1]
    if length == 1:
        return torch.ones_like(negative_lower)
    inverse = _diagonal_blocks(negative_lower, 2) + torch.eye(
        2, dtype=negative_lower.dtype, device=negative_lower.device
    )
    size = 4
    while size <= length:
        blocks = _diagonal_blocks(negative_lower, size)
        half = size // 2
        left, right = inverse[..., 0::2, :, :], inverse[..., 1::2, :, :]
        cross = right @ blocks[..., half:, :half] @ left
        zeros = torch.zeros_like(left)
        top = torch.cat((left, zeros), dim=-1)
        bottom = torch.cat((cross, right), dim=-1)
        inverse = torch.cat((top, bottom), dim=-2)
        size *= 2
    return inverse.squeeze(-3)


def gdn_chunk(query, key, value, g, beta, *, initial_state, qk_l2norm, chunk_size):
    """Parallel intra-chunk GDN with stable block triangular inversion.

    Invert [[A,0],[C,D]] as [[Ai,0],[-Di C Ai,Di]], recursively doubling
    block size. Unlike a global nilpotent power series, this avoids enormous
    cancelling intermediates for correlated keys with near-unit beta.
    """
    if chunk_size < 1 or chunk_size & (chunk_size - 1):
        raise ValueError("chunk_size must be a positive power of two")
    q, k, v, decay, b = _prepare(query, key, value, g, beta, qk_l2norm=qk_l2norm)
    length = q.shape[2]
    padding = (-length) % chunk_size
    q, k, v = [F.pad(x, (0, 0, 0, padding)) for x in (q, k, v)]
    b, decay = [F.pad(x, (0, padding)) for x in (b, decay)]
    shape = (q.shape[0], q.shape[1], -1, chunk_size)
    q, k, v = [x.reshape(*shape, x.shape[-1]) for x in (q, k, v)]
    b, decay = [x.reshape(*shape) for x in (b, decay)]
    cumulative = decay.cumsum(-1)
    upper = torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool).triu(
        1
    )
    lower = torch.ones_like(upper).tril(-1)
    difference = cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)
    pairwise = difference.masked_fill(upper, -float("inf")).exp()
    kb = k * b.unsqueeze(-1)
    vb = v * b.unsqueeze(-1)
    negative_lower = -(kb @ k.transpose(-1, -2) * pairwise).masked_fill(~lower, 0)
    inverse = _unit_lower_inverse(negative_lower)
    new_values = inverse @ vb
    decayed_keys = inverse @ (kb * cumulative.exp().unsqueeze(-1))
    attn = (q @ k.transpose(-1, -2)) * pairwise
    q = q * cumulative.exp().unsqueeze(-1)
    k = k * (cumulative[..., -1:] - cumulative).exp().unsqueeze(-1)
    state = (
        v.new_zeros(v.shape[0], v.shape[1], k.shape[-1], v.shape[-1])
        if initial_state is None
        else initial_state.transpose(-1, -2).contiguous().float()
    )
    outputs = []
    for i in range(q.shape[2]):
        corrected = new_values[:, :, i] - decayed_keys[:, :, i] @ state
        outputs.append(q[:, :, i] @ state + attn[:, :, i] @ corrected)
        state = (
            state * cumulative[:, :, i, -1].exp()[..., None, None]
            + k[:, :, i].transpose(-1, -2) @ corrected
        )
    out = torch.stack(outputs, dim=2).flatten(2, 3)[:, :, :length]
    return (
        out.transpose(1, 2).contiguous().to(query.dtype),
        state.transpose(-1, -2).contiguous(),
    )
