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

"""Spawn one process per chip. This file does not import torch."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/u2601213479/tokenspeed")
PYTHON = os.environ.get("HW2_PYTHON", "/opt/miniconda3/envs/aics/bin/python")
WORKER = Path(__file__).with_name("worker.py")


def main() -> int:
    chips = os.environ.get("HW2_CHIPS", "").split(",")
    chips = [chip.strip() for chip in chips if chip.strip()]
    if not chips:
        print("HW2_CHIPS is required, for example 1,5", file=sys.stderr)
        return 2
    port = os.environ.get("HW2_PORT", "29671")
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = os.pathsep.join(
        [
            str(ROOT / "python"),
            str(ROOT / "tokenspeed-kernel" / "python"),
            str(ROOT / "tokenspeed-kernel-npu" / "python"),
            env_base.get("PYTHONPATH", ""),
        ]
    )
    env_base["MASTER_ADDR"] = "127.0.0.1"
    env_base["MASTER_PORT"] = port
    env_base["WORLD_SIZE"] = str(len(chips))
    processes = []
    logs = []
    log_dir = Path(os.environ.get("HW2_LOG_DIR", "/tmp/hw2-logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    for rank, chip in enumerate(chips):
        env = env_base.copy()
        env["RANK"] = str(rank)
        env["LOCAL_RANK"] = str(rank)
        env["ASCEND_RT_VISIBLE_DEVICES"] = chip
        env["HW2_PHYSICAL_CHIP"] = chip
        log_path = log_dir / f"rank{rank}.log"
        handle = open(log_path, "w")
        logs.append(handle)
        processes.append(
            subprocess.Popen(
                [PYTHON, str(WORKER), *sys.argv[1:]],
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        )
    code = 0
    for process in processes:
        code = max(code, process.wait())
    for handle in logs:
        handle.close()
    for rank in range(len(chips)):
        text = (log_dir / f"rank{rank}.log").read_text(errors="replace")
        print(f"\n===== rank {rank} chip {chips[rank]} =====")
        print(text[-8000:])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
