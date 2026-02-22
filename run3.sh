#!/usr/bin/env bash
set -euo pipefail

WORKER_NUM=${1:-1}
PROCESS_NUM=$((WORKER_NUM + 1))
echo $PROCESS_NUM

hostname > mpi_host_file

CFG="config/simulation/fedml_config.yaml"

# 日志目录（当前目录）
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/fedml_mpi_np${PROCESS_NUM}_${TS}.log"
echo "Logging to: $LOG_FILE"

# 启动 mpirun（后台），输出同时写文件+终端
mpirun -np "$PROCESS_NUM" \
  -hostfile mpi_host_file --oversubscribe \
  python torch_main.py --cf "$CFG" 2>&1 | tee -a "$LOG_FILE" &
PIPE_PID=$!

# 找到实际的 mpirun/prterun PID（用于 TERM）
# 说明：因为用了管道，$! 是 tee 的 PID，不是 mpirun
sleep 0.2
MPIRUN_PID="$(pgrep -n -f "mpirun -np ${PROCESS_NUM} .*torch_main.py --cf ${CFG}" || true)"
if [[ -z "$MPIRUN_PID" ]]; then
  MPIRUN_PID="$(pgrep -n -f "prterun -np ${PROCESS_NUM} .*torch_main.py --cf ${CFG}" || true)"
fi
echo "Launcher PID: ${MPIRUN_PID:-unknown}"

# 监控日志，发现 __finish 就温和结束 launcher
( tail -Fn0 "$LOG_FILE" | grep -m1 "__finish" >/dev/null && \
  echo "Detected __finish -> stopping MPI launcher..." && \
  [[ -n "${MPIRUN_PID:-}" ]] && kill -TERM "$MPIRUN_PID" 2>/dev/null || true ) &
WATCH_PID=$!

# 等待管道结束（等同于等 mpirun 输出结束）
wait "$PIPE_PID" || true

# 收尾：停掉 watcher（如果还在）
kill -TERM "$WATCH_PID" 2>/dev/null || true
wait "$WATCH_PID" 2>/dev/null || true