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

"""Summarize TokenSpeed cache traces and check restored generation."""

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

CAPACITY_TAG = "[host_cache_capacity] "
BLOCK_TAG = "[cache_tiers] "
COPY_TAG = "[host_cache_copy] "
MEASURE_TAG = "[hw3_measurement_begin]"


def read_trace(path):
    lines = path.read_text().splitlines()
    has_marker = any(MEASURE_TAG in line for line in lines)
    decoder = json.JSONDecoder()
    measured = not has_marker
    before, capacity = {}, {}
    blocks, copies, frontiers = [], [], []
    device_bytes = []
    for line in lines:
        if MEASURE_TAG in line:
            measured = True
        if CAPACITY_TAG in line:
            sample, _ = decoder.raw_decode(line.split(CAPACITY_TAG, 1)[1])
            capacity[sample["rank"]] = sample
        if match := re.search(r"Allocated cache arena: device_bytes=(\d+)", line):
            device_bytes.append(int(match[1]))
        if BLOCK_TAG in line:
            sample, _ = decoder.raw_decode(line.split(BLOCK_TAG, 1)[1])
            if measured:
                blocks.append(sample)
            else:
                before[sample["rank"]] = sample
        if measured and COPY_TAG in line:
            sample, _ = decoder.raw_decode(line.split(COPY_TAG, 1)[1])
            copies.append(sample)
        if measured and (
            match := re.search(
                r"(?:retract: suspended|restore:) request (\w+) \((\d+) tokens\)", line
            )
        ):
            frontiers.append(
                (
                    "restore" if "restore:" in line else "retract",
                    match[1],
                    int(match[2]),
                )
            )
    return capacity, before, blocks, copies, frontiers, device_bytes


def summarize(data, trace):
    capacity, before, blocks, copies, frontiers, device_bytes = trace
    ranks = sorted({sample["rank"] for sample in blocks})
    by_rank = {}
    for rank in ranks:
        samples = [sample for sample in blocks if sample["rank"] == rank]
        peak_host_blocks = max(sample["host_used_lcm_blocks"] for sample in samples)
        initial = before.get(rank, {})
        last = samples[-1]
        by_rank[rank] = dict(
            device_total_blocks=last["device_total_lcm_blocks"],
            device_peak_used_blocks=max(
                sample["device_used_lcm_blocks"] for sample in samples
            ),
            device_min_available_blocks=min(
                sample["device_available_lcm_blocks"] for sample in samples
            ),
            host_peak_used_l2_bytes=peak_host_blocks
            * capacity.get(rank, {}).get("host_lcm_block_bytes", 0),
            host_pinned_final=last["host_pinned_blocks"],
            snapshot_free_initial=initial.get(
                "snapshot_free_lcm_blocks", samples[0]["snapshot_free_lcm_blocks"]
            ),
            snapshot_free_final=last["snapshot_free_lcm_blocks"],
            retracted_final=last["retracted_requests"],
        )
    bytes_by_kind = Counter()
    per_op_ms = defaultdict(dict)
    restore_acks = defaultdict(set)
    for copy in copies:
        kind = copy["kind"]
        bytes_by_kind[kind] += copy["page_bytes"]
        per_op_ms[kind][tuple(copy["op_ids"])] = max(
            copy["elapsed_ms"], per_op_ms[kind].get(tuple(copy["op_ids"]), 0)
        )
        if kind == "RestoreDone":
            for op_id in copy["op_ids"]:
                restore_acks[op_id].add(copy["rank"])
    median_ms = {
        kind: statistics.median(samples.values()) for kind, samples in per_op_ms.items()
    }
    final = next((sample for sample in reversed(blocks) if sample["rank"] == 0), {})
    initial = before.get(0, {})
    request_ids = {row["request_id"] for row in data["requests"]}
    prefill_tokens = sum(
        length
        for sample in blocks
        if sample["rank"] == 0
        for rid, length, _ in zip(
            sample.get("request_ids", []),
            sample.get("input_lengths", []),
            sample.get("extend_prefix_lens", []),
        )
        if rid in request_ids
    )
    return dict(
        completed=data["completed"],
        requests=len(data["requests"]),
        tokens_per_s=data["output_tokens_per_s"],
        median_ttft_ms=data["median_ttft_ms"],
        prefill_tokens_computed=prefill_tokens,
        scheduler_rss_bytes=data["scheduler_rss_bytes"],
        device_cache_reserved_bytes=sum(device_bytes),
        host_capacity=capacity,
        blocks_by_rank=by_rank,
        host_hits=final.get("host_prefix_hits", 0) - initial.get("host_prefix_hits", 0),
        host_misses=final.get("host_prefix_misses", 0)
        - initial.get("host_prefix_misses", 0),
        host_evictions=final.get("host_evictions", 0)
        - initial.get("host_evictions", 0),
        d2h_bytes=bytes_by_kind["WriteBackDone"] + bytes_by_kind["SnapshotDone"],
        h2d_bytes=bytes_by_kind["LoadBackDone"] + bytes_by_kind["RestoreDone"],
        transfer_median_ms=median_ms,
        acknowledged_restores=len(restore_acks),
        all_ranks_acknowledged_restores=sum(
            acks == set(ranks) for acks in restore_acks.values()
        ),
        restore_ranks=sorted(
            {copy["rank"] for copy in copies if copy["kind"] == "RestoreDone"}
        ),
        frontiers=sorted(
            {(kind, rid, count) for kind, rid, count in frontiers if rid in request_ids}
        ),
    )


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs, summaries = {}, {}
    for result in sorted(args.results.glob("*.json")):
        log_path = args.logs / f"{result.stem}.log"
        if not log_path.exists():
            continue
        data = json.loads(result.read_text())
        runs[result.stem] = data
        summaries[result.stem] = summarize(data, read_trace(log_path))

    for name, data in runs.items():
        if name.endswith("-resume"):
            reference = runs.get(name.removesuffix("-resume") + "-reference")
        elif name.endswith(("-small", "-large", "-larger")):
            reference = runs.get(name.rsplit("-", 1)[0] + "-off")
        else:
            reference = None
        if reference is not None:
            pairs = list(zip(data["requests"], reference["requests"], strict=True))
            summaries[name]["matches_reference_tokens"] = [
                observed["token_ids"] == expected["token_ids"]
                for observed, expected in pairs
            ]
            if any(observed["logprobs"] for observed, _ in pairs):
                summaries[name]["max_sampled_logprob_delta"] = max(
                    abs(actual[0] - expected[0])
                    for run, reference_run in pairs
                    for actual, expected in zip(
                        run["logprobs"], reference_run["logprobs"], strict=True
                    )
                )
    args.output.write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    run()
