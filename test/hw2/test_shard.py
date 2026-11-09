"""CPU checks for TP slicing. No NPU and no checkpoint."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

from tokenspeed.runtime.models.qwen3_5_npu_parallel import grouped_slice, require_divisible


class ShardTests(unittest.TestCase):
    def test_one_group_splits_into_contiguous_halves(self):
        # 8 query heads, each head is 2 * head_dim = 512 rows. TP=2 keeps 4 whole heads.
        weight = torch.arange(8 * 512 * 4, dtype=torch.float32).reshape(8 * 512, 4)
        rank0 = grouped_slice(weight, [8 * 512], 0, 2, dim=0)
        rank1 = grouped_slice(weight, [8 * 512], 1, 2, dim=0)
        self.assertEqual(tuple(rank0.shape), (4 * 512, 4))
        self.assertEqual(tuple(rank1.shape), (4 * 512, 4))
        self.assertTrue(torch.equal(torch.cat((rank0, rank1), dim=0), weight))
        self.assertEqual(int(rank0.shape[0] % 512), 0)

    def test_component_groups_keep_each_block_intact(self):
        # q, k, v are separate groups. Rank 0 holds the first half of each, not a mid-head cut.
        groups = [8, 8, 8]
        weight = torch.arange(24 * 3, dtype=torch.float32).reshape(24, 3)
        rank0 = grouped_slice(weight, groups, 0, 2, dim=0)
        self.assertEqual(tuple(rank0.shape), (12, 3))
        self.assertTrue(torch.equal(rank0[:4], weight[:4]))
        self.assertTrue(torch.equal(rank0[4:8], weight[8:12]))
        self.assertTrue(torch.equal(rank0[8:], weight[16:20]))

    def test_column_shard(self):
        weight = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
        rank0 = grouped_slice(weight, [8], 0, 2, dim=1)
        rank1 = grouped_slice(weight, [8], 1, 2, dim=1)
        self.assertEqual(tuple(rank0.shape), (4, 4))
        self.assertTrue(torch.equal(torch.cat((rank0, rank1), dim=1), weight))

    def test_refuse_indivisible_heads(self):
        config = SimpleNamespace(
            num_attention_heads=8,
            num_key_value_heads=3,
            linear_num_key_heads=16,
            linear_num_value_heads=16,
            intermediate_size=3584,
        )
        with self.assertRaises(RuntimeError):
            require_divisible(config, 2)

    def test_divisible_config_is_accepted(self):
        config = SimpleNamespace(
            num_attention_heads=24,
            num_key_value_heads=4,
            linear_num_key_heads=16,
            linear_num_value_heads=48,
            intermediate_size=17408,
        )
        require_divisible(config, 2)


if __name__ == "__main__":
    unittest.main()
