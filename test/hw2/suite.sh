#!/bin/bash
set -u
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONPATH="/home/u2601213479/tokenspeed/python:/home/u2601213479/tokenspeed/tokenspeed-kernel/python:/home/u2601213479/tokenspeed/tokenspeed-kernel-npu/python:${PYTHONPATH:-}"
PY=/opt/miniconda3/envs/aics/bin/python
OUT="$(cd "$(dirname "$0")/../.." && pwd)/hw2_results"
LAUNCH=/home/u2601213479/tokenspeed/test/hw2/launch.py
WORKER=/home/u2601213479/tokenspeed/test/hw2/worker.py
mkdir -p "$OUT"
PROMPT="请用一句话介绍北京大学。"

SUITE_FAIL=0

run_tp1() {
  local name="$1"; shift
  echo "RUN $name tp1 $(date)"
  if ! ASCEND_RT_VISIBLE_DEVICES=1 RANK=0 WORLD_SIZE=1 HW2_PHYSICAL_CHIP=1 \
    "$PY" "$WORKER" --output "$OUT/$name.json" "$@" \
    > "$OUT/$name.log" 2>&1; then
    echo "FAIL $name" | tee -a "$OUT/suite.log"
    SUITE_FAIL=1
    return 1
  fi
}

run_tp2() {
  local name="$1"; shift
  local port="$1"; shift
  echo "RUN $name tp2 $(date)"
  if ! HW2_CHIPS=4,5 HW2_PORT="$port" HW2_LOG_DIR="$OUT/logs-$name" \
    "$PY" "$LAUNCH" --output "$OUT/$name.json" "$@" \
    > "$OUT/$name.log" 2>&1; then
    echo "FAIL $name" | tee -a "$OUT/suite.log"
    SUITE_FAIL=1
    return 1
  fi
}

model08=/home/u2601213479/models/Qwen3.5-0.8B
for rep in 0 1 2; do
  for seq in 32 512; do
    for batch in 1 4; do
      run_tp1 "m08_tp1_s${seq}_b${batch}_r${rep}" --model-dir "$model08" --mode perf --steps 16 --seq-len "$seq" --batch "$batch" --prompt "$PROMPT"
      run_tp2 "m08_tp2_s${seq}_b${batch}_r${rep}" $((29700 + rep * 10 + seq + batch)) --model-dir "$model08" --mode perf --steps 16 --seq-len "$seq" --batch "$batch" --prompt "$PROMPT"
    done
  done
done

# overlap versus synchronous allreduce, same shape
for seq in 32 512; do
  for batch in 1 4; do
    run_tp2 "m08_sync_s${seq}_b${batch}" $((29810 + seq + batch)) --model-dir "$model08" --mode perf --steps 8 --seq-len "$seq" --batch "$batch" --prompt "$PROMPT"
    run_tp2 "m08_overlap_s${seq}_b${batch}" $((29840 + seq + batch)) --model-dir "$model08" --mode perf --steps 8 --seq-len "$seq" --batch "$batch" --prompt "$PROMPT" --overlap
  done
done

# natural prompts
run_tp2 "m08_demo_pku" 29880 --model-dir "$model08" --mode smoke --steps 32 --prompt "请用一句话介绍北京大学。"
run_tp2 "m08_demo_math" 29881 --model-dir "$model08" --mode smoke --steps 16 --prompt "计算 12 乘以 13，只给出结果。"

run_tp2 "m08_isolate" 29890 --model-dir "$model08" --mode isolate --prompt "$PROMPT"
echo SUITE_08_DONE $(date) | tee -a "$OUT/suite.log"

wait_model() {
  local dir="$1"
  echo "WAIT $dir $(date)"
  for _ in $(seq 1 240); do
    if [ -f "$dir/model.safetensors.index.json" ] && [ -f "$dir/config.json" ] && ! ls "$dir"/*.partial >/dev/null 2>&1; then
      local expected
      expected=$("$PY" -c "import json;print(len(set(json.load(open('$dir/model.safetensors.index.json'))['weight_map'].values())))")
      local have
      have=$(ls "$dir"/model.safetensors-*.safetensors 2>/dev/null | wc -l)
      if [ "$have" -ge "$expected" ]; then
        echo "READY $dir shards=$have"
        return 0
      fi
    fi
    sleep 30
  done
  echo "TIMEOUT $dir" | tee -a "$OUT/suite.log"
  return 1
}

run_family() {
  local size="$1"
  local dir="/home/u2601213479/models/Qwen3.5-${size}"
  wait_model "$dir" || return 0
  run_tp2 "m${size}_demo" $((29900)) --model-dir "$dir" --mode smoke --steps 8 --prompt "$PROMPT"
  run_tp1 "m${size}_tp1_try" --model-dir "$dir" --mode logits --steps 4 --seq-len 32 --batch 1 --prompt "$PROMPT" || true
  run_tp2 "m${size}_tp2_logits" $((29910)) --model-dir "$dir" --mode logits --steps 4 --seq-len 32 --batch 1 --prompt "$PROMPT"
  run_tp2 "m${size}_s32_b1" $((29920)) --model-dir "$dir" --mode perf --steps 8 --seq-len 32 --batch 1 --prompt "$PROMPT"
  run_tp2 "m${size}_s128_b1" $((29930)) --model-dir "$dir" --mode perf --steps 8 --seq-len 128 --batch 1 --prompt "$PROMPT"
  run_tp2 "m${size}_s32_b2" $((29940)) --model-dir "$dir" --mode perf --steps 8 --seq-len 32 --batch 2 --prompt "$PROMPT"
}

run_family 4B
run_family 9B
run_family 27B
if [ "$SUITE_FAIL" -ne 0 ]; then
  echo "SUITE_FAILED $(date)" | tee -a "$OUT/suite.log"
  exit 1
fi
echo SUITE_ALL_DONE $(date) | tee -a "$OUT/suite.log"
