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

"""Explicit Ascend boundary for the single-request Qwen3.5 adapter."""

import torch_npu
from tokenspeed_kernel_npu.ops.gdn import causal_conv, gdn_chunk, gdn_recurrent
from tokenspeed_kernel_npu.ops.layernorm import rmsnorm
from tokenspeed_kernel_npu.ops.mha import mha_prefill


def mha_cached(q, k, v, *, mask, scale):
    """Fused attention over dense BNSD cached K/V, including chunk extension."""
    output, _ = torch_npu.npu_fused_infer_attention_score(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        atten_mask=mask,
        num_heads=q.shape[1],
        num_key_value_heads=k.shape[1],
        input_layout="BNSD",
        scale=scale,
        sparse_mode=0,
    )
    return output


__all__ = [
    "causal_conv",
    "gdn_chunk",
    "gdn_recurrent",
    "rmsnorm",
    "mha_prefill",
    "mha_cached",
]
