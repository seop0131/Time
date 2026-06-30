#!/usr/bin/env bash
# 범용 2-GPU 분할 러너.
# 사용법: bash scripts/run_dual_gpu.sh <jobfile>
#   <jobfile> = 한 줄에 하나씩, train.py 뒤에 붙일 인자 전체(따옴표 없이).
# 각 줄을 라운드로빈으로 GPU0/GPU1에 분배 → 두 GPU 동시, 각자 자기 몫 순차 실행.
# PY 환경변수로 python 경로 override 가능.
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
PY="${PY:-/home/seop/.conda/envs/open-mmlab/bin/python}"
PYSCRIPT="${PYSCRIPT:-train.py}"   # 대상 스크립트 (train.py | train_with_synth.py)
JOBFILE="$1"

# 짝/홀 줄로 분배
gpu_worker() {
  local gpu="$1"; local parity="$2"; local i=0
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in \#*) continue ;; esac
    if [ $((i % 2)) -eq "$parity" ]; then
      echo "[GPU${gpu}] >>> ${line}"
      CUDA_VISIBLE_DEVICES="$gpu" ${PY} ${PYSCRIPT} ${line}
      echo "[GPU${gpu}] done: ${line}"
    fi
    i=$((i + 1))
  done < "$JOBFILE"
}

gpu_worker 0 0 &
PID0=$!
gpu_worker 1 1 &
PID1=$!
wait $PID0
wait $PID1
echo "===== dual-GPU run 완료 ====="
