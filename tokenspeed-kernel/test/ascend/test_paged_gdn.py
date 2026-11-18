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

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="requires Ascend NPU",
)


def test_packed_gdn_prefill_and_decode_use_independent_state_rows():
    from tokenspeed_kernel.ops.attention.gdn import gdn_chunk_prefill, gdn_decode_step
    from tokenspeed_kernel.ops.attention.qwen35.ascend import gdn_recurrent

    generator = torch.Generator().manual_seed(123)
    q = torch.randn((1, 75, 2, 16), generator=generator).bfloat16()
    k = torch.randn(q.shape, generator=generator).bfloat16()
    v = torch.randn((1, 75, 4, 16), generator=generator).bfloat16()
    g = -torch.rand((1, 75, 4), generator=generator)
    beta = torch.rand(g.shape, generator=generator)
    initial = torch.randn((2, 4, 16, 16), generator=generator) * 0.1
    result = gdn_chunk_prefill(
        q.npu(),
        k.npu(),
        v.npu(),
        g.npu(),
        beta.npu(),
        scale=0.25,
        initial_state=initial.npu(),
        cu_seqlens=torch.tensor([0, 6, 75], device="npu", dtype=torch.int32),
        qk_l2norm=True,
    )
    for row, (begin, end) in enumerate(((0, 6), (6, 75))):
        out, state = gdn_recurrent(
            q[:, begin:end].repeat_interleave(2, dim=-2),
            k[:, begin:end].repeat_interleave(2, dim=-2),
            v[:, begin:end],
            g[:, begin:end],
            beta[:, begin:end],
            initial_state=initial[row : row + 1],
            qk_l2norm=True,
            scale=0.25,
        )
        torch.testing.assert_close(
            result.out[:, begin:end].cpu(), out, atol=2e-3, rtol=2e-2
        )
        torch.testing.assert_close(
            result.final_state[row : row + 1].cpu(), state, atol=3e-4, rtol=3e-3
        )

    # Rebuild the working rows at different physical addresses and include a
    # padding row to exercise the same slot contract as the runtime.
    pool = torch.zeros((6, 4, 16, 16), device="npu")
    pool[1:3].copy_(result.final_state)
    read_rows = torch.tensor([1, 2, -1], device="npu", dtype=torch.int32)
    write_rows = torch.tensor([4, 5, -1], device="npu", dtype=torch.int32)
    a = torch.randn((3, 1, 4), generator=generator)
    b = torch.randn(a.shape, generator=generator)
    q1, k1, v1 = (x[:, :3].transpose(0, 1).contiguous() for x in (q, k, v))
    decay_log = torch.zeros(4)
    bias = torch.zeros(4)
    out = gdn_decode_step(
        q1.npu(),
        k1.npu(),
        v1.npu(),
        A_log=decay_log.npu(),
        a=a.npu(),
        dt_bias=bias.npu(),
        b=b.npu(),
        initial_state=pool,
        initial_state_indices=read_rows,
        output_state_indices=write_rows,
        scale=0.25,
    )
    expected_out, expected_state = gdn_recurrent(
        q1[:2].repeat_interleave(2, dim=-2),
        k1[:2].repeat_interleave(2, dim=-2),
        v1[:2],
        -F.softplus(a[:2]),
        b[:2].sigmoid(),
        initial_state=result.final_state.cpu(),
        qk_l2norm=True,
        scale=0.25,
    )
    torch.testing.assert_close(out[:2].cpu(), expected_out, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(pool[4:6].cpu(), expected_state, atol=3e-4, rtol=3e-3)
    assert torch.equal(pool[1:3].cpu(), result.final_state.cpu())
    assert torch.count_nonzero(out[2]).item() == 0
    assert torch.count_nonzero(pool[0]).item() == 0


def test_paged_convolution_uses_only_its_own_restored_history():
    from tokenspeed_kernel.ops.attention.qwen35.ascend import (
        causal_conv,
        causal_conv_paged_prefill,
        causal_conv_paged_update,
    )

    generator = torch.Generator().manual_seed(12)
    data = torch.randn((16, 9), generator=generator).bfloat16().npu()
    weights = torch.randn((16, 4), generator=generator).bfloat16().npu()
    pool = torch.zeros((6, 3, 16), dtype=torch.bfloat16, device="npu").transpose(1, 2)
    pool[1].copy_(data[:, :3])
    pool[2].fill_(5)
    reference_warm, state_warm = causal_conv(
        data[None, :, 3:6],
        weights,
        None,
        state=F.pad(data[None, :, :3], (1, 0)),
        activation="silu",
    )
    reference_cold, state_cold = causal_conv(
        data[None, :, 6:],
        weights,
        None,
        state=None,
        activation="silu",
    )
    actual = causal_conv_paged_prefill(
        data[:, 3:],
        weights,
        None,
        conv_states=pool,
        boundaries=torch.tensor([0, 3, 6], device="npu"),
        cache_indices=torch.tensor([1, 2], device="npu"),
        has_initial_state=torch.tensor([True, False], device="npu"),
        activation="silu",
        pad_slot_id=-1,
    )
    assert torch.equal(
        actual.cpu(), torch.cat((reference_warm[0], reference_cold[0]), dim=-1).cpu()
    )
    next_token = data[None, :, :1].expand(2, -1, -1).contiguous()
    expected, next_state = causal_conv(
        next_token,
        weights,
        None,
        state=torch.cat((state_warm, state_cold)),
        activation="silu",
    )
    actual = causal_conv_paged_update(
        next_token,
        weights,
        None,
        conv_states=pool,
        read_rows=torch.tensor([1, 2], device="npu"),
        write_rows=torch.tensor([4, 5], device="npu"),
        activation="silu",
        pad_slot_id=-1,
    )
    assert torch.equal(actual.cpu(), expected.cpu())
    assert torch.equal(pool[4:6].cpu(), next_state[:, :, 1:].cpu())
    assert torch.equal(
        pool[1:3].cpu(), torch.cat((state_warm, state_cold))[:, :, 1:].cpu()
    )
