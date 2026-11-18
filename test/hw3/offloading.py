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

"""Run Qwen3.5 cache experiments through the TokenSpeed scheduler."""

import argparse
import json
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil

from tokenspeed.runtime.engine.io_struct import GenerateReqInput
from tokenspeed.runtime.entrypoints.engine import Engine


def run_request(engine, label, prompt, steps, *, logprobs, first_token=None):
    request = GenerateReqInput(
        input_ids=prompt,
        user_rid=label,
        sampling_params={"temperature": 0, "max_new_tokens": steps, "ignore_eos": True},
        return_logprob=logprobs,
        stream=True,
    )
    tokens, output_logprobs, arrival_ms = [], [], []
    meta = {}
    start = time.perf_counter()
    try:
        for part in engine.llm.generate_stream(request):
            meta = part.get("meta_info", {})
            new_tokens = part.get("output_ids", [])
            if new_tokens:
                output_logprobs.extend(
                    meta.get("output_token_logprobs", [])[-len(new_tokens) :]
                )
                tokens.extend(new_tokens)
                arrival_ms.append((time.perf_counter() - start) * 1000)
                if first_token is not None:
                    first_token.set()
    finally:
        if first_token is not None:
            first_token.set()
    return dict(
        label=label,
        request_id=meta.get("id"),
        prompt_tokens=len(prompt),
        cached_tokens=meta.get("cached_tokens"),
        finish_reason=meta.get("finish_reason"),
        token_ids=tokens,
        logprobs=output_logprobs,
        ttft_ms=arrival_ms[0] if arrival_ms else None,
        latency_ms=(time.perf_counter() - start) * 1000,
        token_arrival_ms=arrival_ms,
    )


def make_prompt(seed, size):
    rng = random.Random(seed)
    return [rng.randrange(200, 12000) for _ in range(size)]


def run_workload(engine, args):
    prompt_a = make_prompt(10, args.prompt_tokens)
    prompt_b = prompt_a[: args.prompt_tokens // 2] + make_prompt(
        20, len(prompt_a) - args.prompt_tokens // 2
    )
    if args.case == "smoke":
        return [run_request(engine, "A", prompt_a, args.steps, logprobs=args.logprobs)]
    if args.case == "reference":
        return [
            run_request(engine, label, prompt, args.steps, logprobs=args.logprobs)
            for label, prompt in (("A", prompt_a), ("B", prompt_b))
        ]
    if args.case == "resume":
        a_started = threading.Event()
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(
                run_request,
                engine,
                "A",
                prompt_a,
                args.steps,
                logprobs=args.logprobs,
                first_token=a_started,
            )
            if not a_started.wait(timeout=120):
                raise TimeoutError("A did not produce its first token")
            b = pool.submit(
                run_request, engine, "B", prompt_b, args.steps, logprobs=args.logprobs
            )
            return [a.result(), b.result()]

    return [
        run_request(
            engine,
            f"request-{index}",
            make_prompt(
                100 + (index % args.locality if args.case == "high" else index),
                args.prompt_tokens,
            ),
            args.steps,
            logprobs=args.logprobs,
        )
        for index in range(args.requests)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--case", choices=("smoke", "reference", "resume", "high", "low"), required=True
    )
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--device-tokens", type=int, required=True)
    parser.add_argument("--host-ratio", type=float, required=True)
    parser.add_argument("--snapshot-ratio", type=float, required=True)
    parser.add_argument("--force-retraction-interval", type=int, default=0)
    parser.add_argument("--prompt-tokens", type=int, default=260)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--locality", type=int, default=4)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--logprobs", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.host_ratio < 0 or args.snapshot_ratio < 0:
        parser.error("Host ratios must be nonnegative")
    if args.force_retraction_interval and not args.snapshot_ratio:
        parser.error("forced retraction requires a positive snapshot ratio")
    if args.case == "high" and args.prompt_tokens <= 4:
        parser.error(
            "the high-reuse case needs a nonempty prefix and a four-token suffix"
        )

    with Engine(
        model=str(args.model),
        device="npu",
        language_model_only=True,
        attn_tp_size=args.tp,
        port=args.port,
        dtype="bfloat16",
        attention_backend="mha",
        sampling_backend="greedy",
        enforce_eager=True,
        disable_prefill_graph=True,
        disable_pdl=True,
        disable_autotune=True,
        disable_overlap_schedule=True,
        max_model_len=args.prompt_tokens + args.steps + 32,
        max_total_tokens=args.device_tokens,
        max_num_seqs=2,
        chunked_prefill_size=128,
        prefix_granularity=128,
        disable_kvstore=args.host_ratio == 0,
        kvstore_ratio=args.host_ratio,
        kvstore_io_backend="direct",
        retraction_snapshot_ratio=args.snapshot_ratio,
        debug_force_retraction_interval=args.force_retraction_interval,
        skip_tokenizer_init=True,
        stream_output=True,
        enable_output_logprobs=args.logprobs,
        log_level="info",
        seed=1234,
    ) as engine:
        run_request(engine, "warmup", make_prompt(0, 65), 2, logprobs=args.logprobs)
        if args.case == "high":
            for index in range(args.locality):
                run_request(
                    engine,
                    f"prime-{index}",
                    make_prompt(100 + index, args.prompt_tokens)[:-4],
                    2,
                    logprobs=args.logprobs,
                )
            time.sleep(0.1)
        print("[hw3_measurement_begin]", flush=True)
        start = time.perf_counter()
        requests = run_workload(engine, args)
        duration = time.perf_counter() - start
        scheduler_rss = sum(
            proc.memory_info().rss
            for proc in psutil.Process().children(recursive=True)
            if proc.is_running() and proc.name().startswith("tokenspeedsch")
        )

    ttft = [row["ttft_ms"] for row in requests if row["ttft_ms"] is not None]
    result = dict(
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        duration_s=duration,
        median_ttft_ms=statistics.median(ttft) if ttft else None,
        output_tokens_per_s=sum(len(row["token_ids"]) for row in requests) / duration,
        completed=sum(len(row["token_ids"]) == args.steps for row in requests),
        scheduler_rss_bytes=scheduler_rss,
        requests=requests,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
