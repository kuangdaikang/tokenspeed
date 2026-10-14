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

"""Numerical and state tests against unmodified Transformers CPU functions."""

import json
import unittest

import torch
import torch.nn.functional as F
import torch_npu
from tokenspeed_kernel.ops.attention.qwen35.ascend import (
    causal_conv,
    gdn_chunk,
    gdn_recurrent,
    mha_prefill,
    rmsnorm,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_recurrent_gated_delta_rule,
)

RESULTS = []


def compare(actual, expected, label, *, atol, rtol):
    actual, expected = actual.cpu().float(), expected.cpu().float()
    error = actual - expected
    record = dict(
        label=label,
        max_abs=error.abs().max().item(),
        rmse=error.square().mean().sqrt().item(),
        atol=atol,
        rtol=rtol,
    )
    RESULTS.append(record)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


class GdnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(16)
        torch.npu.set_device(0)
        torch.npu.set_compile_mode(jit_compile=False)
        torch.manual_seed(20261014)

    def test_gdn_shape_boundaries_and_nonzero_state(self):
        for length, heads, dim in [
            (1, 4, 32),
            (3, 4, 32),
            (63, 4, 32),
            (64, 16, 128),
            (65, 16, 128),
            (129, 4, 32),
            (512, 16, 128),
        ]:
            with self.subTest(length=length, heads=heads, dim=dim):
                q, k, v = [
                    torch.randn(1, length, heads, dim).bfloat16() for _ in range(3)
                ]
                g = -torch.rand(1, length, heads)
                beta = torch.rand(1, length, heads).bfloat16()
                state = torch.randn(1, heads, dim, dim) * 0.02
                expected, final = torch_recurrent_gated_delta_rule(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state=state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
                arguments = [t.to("npu:0") for t in (q, k, v, g, beta)]
                initial = state.transpose(-1, -2).contiguous().to("npu:0")
                for name, fn in [("chunk", gdn_chunk), ("recurrent", gdn_recurrent)]:
                    kwargs = dict(initial_state=initial, qk_l2norm=True)
                    if name == "chunk":
                        kwargs["chunk_size"] = 64
                    actual, updated = fn(*arguments, **kwargs)
                    compare(
                        actual,
                        expected,
                        f"gdn_{name}_T{length}_H{heads}_D{dim}",
                        atol=2e-3,
                        rtol=1e-2,
                    )
                    compare(
                        updated.transpose(-1, -2),
                        final,
                        f"state_{name}_T{length}",
                        atol=2e-5,
                        rtol=2e-4,
                    )

    def test_correlated_keys_unit_beta(self):
        # A global matrix power series is unstable on this learned-like case.
        for length in (64, 65, 512):
            key = (
                torch.randn(1, 1, 4, 32)
                .bfloat16()
                .expand(1, length, 4, 32)
                .contiguous()
            )
            query = key.clone()
            value = torch.randn(1, length, 4, 32).bfloat16()
            g = torch.zeros(1, length, 4)
            beta = torch.ones(1, length, 4).bfloat16()
            expected, final = torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            actual, updated = gdn_chunk(
                *[t.to("npu:0") for t in (query, key, value, g, beta)],
                initial_state=None,
                qk_l2norm=True,
                chunk_size=64,
            )
            compare(
                actual, expected, f"correlated_keys_T{length}", atol=2e-3, rtol=1e-2
            )
            compare(
                updated.transpose(-1, -2),
                final,
                f"correlated_state_T{length}",
                atol=2e-5,
                rtol=2e-4,
            )

    def test_gdn_split_state_updates(self):
        length = 73
        q, k, v = [torch.randn(1, length, 16, 128).bfloat16() for _ in range(3)]
        g = -torch.rand(1, length, 16)
        b = torch.rand(1, length, 16).bfloat16()
        data = [t.to("npu:0") for t in (q, k, v, g, b)]
        whole, final = gdn_chunk(
            *data, initial_state=None, qk_l2norm=True, chunk_size=64
        )
        first, state = gdn_chunk(
            *[t[:, :65] for t in data],
            initial_state=None,
            qk_l2norm=True,
            chunk_size=64,
        )
        outputs = [first]
        for i in range(65, length):
            out, state = gdn_recurrent(
                *[t[:, i : i + 1] for t in data], initial_state=state, qk_l2norm=True
            )
            outputs.append(out)
        compare(
            torch.cat(outputs, 1), whole, "gdn_prefill65_decode8", atol=2e-3, rtol=1e-2
        )
        compare(state, final, "gdn_multistep_final_state", atol=2e-5, rtol=2e-4)

    def test_conv_boundaries_and_split(self):
        for length in (1, 3, 4, 65):
            channels = 6144 if length == 65 else 16
            x = torch.randn(1, channels, length).bfloat16()
            w = (torch.randn(channels, 4) * 0.2).bfloat16()
            expected = F.silu(
                F.conv1d(x.float(), w.float().unsqueeze(1), padding=3, groups=channels)[
                    :, :, :length
                ].bfloat16()
            )
            actual, state = causal_conv(
                x.to("npu:0"), w.to("npu:0"), None, state=None, activation="silu"
            )
            compare(
                actual, expected, f"conv_T{length}_C{channels}", atol=1e-2, rtol=1e-2
            )
            pieces = []
            split_state = None
            for i in range(length):
                out, split_state = causal_conv(
                    x[:, :, i : i + 1].to("npu:0"),
                    w.to("npu:0"),
                    None,
                    state=split_state,
                    activation="silu",
                )
                pieces.append(out)
            compare(
                torch.cat(pieces, -1), actual, f"conv_split_T{length}", atol=0, rtol=0
            )
            compare(split_state, state, f"conv_state_T{length}", atol=0, rtol=0)

    def test_rmsnorm_and_attention(self):
        x = torch.randn(65, 1024)
        weight = torch.randn(1024) * 0.1 + 1
        expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight
        actual = rmsnorm(x.to("npu:0"), weight.to("npu:0"), 1e-6)
        compare(actual, expected, "rmsnorm_fp32", atol=2e-5, rtol=2e-5)
        for length in (1, 17, 65):
            q = torch.randn(length, 8, 256).bfloat16()
            k, v = [torch.randn(length, 2, 256).bfloat16() for _ in range(2)]
            scores = (
                q.transpose(0, 1).float()
                @ k.repeat_interleave(4, dim=1)
                .transpose(0, 1)
                .float()
                .transpose(-1, -2)
            ) / 16
            scores = scores.masked_fill(
                torch.ones(length, length, dtype=torch.bool).triu(1), -float("inf")
            )
            expected = (
                (
                    scores.softmax(-1)
                    @ v.repeat_interleave(4, dim=1).transpose(0, 1).float()
                )
                .transpose(0, 1)
                .bfloat16()
            )
            out = mha_prefill(
                *[t.to("npu:0") for t in (q, k, v)], None, [0, length], length
            )
            compare(out, expected, f"gqa_T{length}_H8_Hkv2_D256", atol=2e-2, rtol=2e-2)

    def test_packed_gqa_head256(self):
        q = torch.randn(82, 8, 256).bfloat16()
        k, v = [torch.randn(82, 2, 256).bfloat16() for _ in range(2)]
        device = [t.to("npu:0") for t in (q, k, v)]
        packed = mha_prefill(*device, None, [0, 17, 82], 65)
        single = torch.cat(
            [
                mha_prefill(*[t[a:b] for t in device], None, [0, b - a], b - a)
                for a, b in [(0, 17), (17, 82)]
            ]
        )
        compare(packed, single, "packed_gqa_head256", atol=0, rtol=0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(GdnTests)
    )
    with open(args.output, "w") as f:
        json.dump(
            dict(
                passed=result.wasSuccessful(),
                tests=result.testsRun,
                comparisons=RESULTS,
            ),
            f,
            indent=2,
        )
    raise SystemExit(0 if result.wasSuccessful() else 1)
