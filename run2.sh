#!/usr/bin/env bash
set -euo pipefail

WORKER_NUM=${1:-1}
PROCESS_NUM=$((WORKER_NUM + 1))
echo $PROCESS_NUM

hostname > mpi_host_file
CFG="config/simulation/fedml_config.yaml"

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/fedml_mpi_np${PROCESS_NUM}_${TS}.log"
echo "Logging to: $LOG_FILE"

# 后台跑 mpirun，本体 PID 就是 mpirun
mpirun -np $PROCESS_NUM -hostfile mpi_host_file --oversubscribe \
  python torch_main.py --cf "${CFG}" \
  >>"$LOG_FILE" 2>&1 &
MPIRUN_PID=$!

cleanup() {
  # 只有在被中断/kill 时才清理（不在正常退出时乱动）
  echo "Cleaning up..."
  kill -TERM "$MPIRUN_PID" 2>/dev/null || true
  sleep 1
  kill -KILL "$MPIRUN_PID" 2>/dev/null || true
}
trap cleanup INT TERM

wait "$MPIRUN_PID"