#!/bin/bash

LOG_DIR="trainlog2"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/master_queue_$(date +%Y%m%d_%H%M%S).log"
RUN_LOCK="/tmp/fed_task_running.lock"

CONFIG_LIST=(
  # "config/20newsC4/fedml_config_compare_llama_24g.yaml"
  # "config/20newsC4/fedml_config_compare_llamaprox_24g.yaml"
  # "config/20newsC4/fedml_config_compare_mor_expert_24g.yaml"
  # "config/20newsC4/fedml_config_compare_mor_expert_adaptive_warmup_24g.yaml"
  #   "config/20newsC6/fedml_config_compare_llama_24g.yaml"
  # "config/20newsC6/fedml_config_compare_llamaprox_24g.yaml"
  # "config/20newsC6/fedml_config_compare_mor_expert_24g.yaml"
  # "config/20newsC6/fedml_config_compare_mor_expert_adaptive_warmup_24g.yaml"
  #   "config/20newsC10/fedml_config_compare_llama_24g.yaml"
  # "config/20newsC10/fedml_config_compare_llamaprox_24g.yaml"
  # "config/20newsC10/fedml_config_compare_mor_expert_24g.yaml"
  # "config/20newsC10/fedml_config_compare_mor_expert_adaptive_warmup_24g.yaml"
  # "config/MoR_0.7/fedml_config_compare_mor_expert_24g.yaml"
  # "config/MoR_0.7/fedml_config_compare_mor_expert_adaptive_warmup_24g.yaml"
  # "config/MoR_0.9/fedml_config_compare_mor_expert_24g.yaml"
  # "config/MoR_0.9/fedml_config_compare_mor_expert_adaptive_warmup_24g.yaml"
  # "config/MoR_1/fedml_config_compare_mor_expert_24g.yaml"
  "config/20news/fedml_config_compare_mor_expert_adaptive_warmup_24g.yaml"
  "config/agnews/fedml_config_compare_mor_expert_adaptive_warmup_24g.yaml"
  "config/simulation/fedml_config_compare_mor_expert_adaptive_warmup_24g.yaml"
)

echo "[$(date '+%F %T')] 后台串行任务启动" >> "$MASTER_LOG"

for CONFIG_FILE in "${CONFIG_LIST[@]}"; do
  BASENAME=$(basename "$CONFIG_FILE" .yaml)
  TASK_LOG="$LOG_DIR/${BASENAME}_$(date +%Y%m%d_%H%M%S).log"

  echo "[$(date '+%F %T')] 准备执行: $CONFIG_FILE" >> "$MASTER_LOG"

  while [ -f "$RUN_LOCK" ]; do
    LOCK_PID=$(cat "$RUN_LOCK" 2>/dev/null)
    if [ -n "$LOCK_PID" ] && kill -0 "$LOCK_PID" 2>/dev/null; then
      echo "[$(date '+%F %T')] 前一任务仍在运行，PID=$LOCK_PID，等待 30s" >> "$MASTER_LOG"
      sleep 30
    else
      echo "[$(date '+%F %T')] 检测到残留锁，自动清理" >> "$MASTER_LOG"
      rm -f "$RUN_LOCK"
    fi
  done

  (
    echo $$ > "$RUN_LOCK"

    cleanup() {
      rm -f "$RUN_LOCK"
    }
    trap cleanup EXIT

    {
      echo "--- TASK START: $(date) ---"
      echo "--- CONFIG FILE: $CONFIG_FILE ---"
      echo "--- CONFIG CONTENT ---"
      cat "$CONFIG_FILE"
      echo
      echo "--- COMMAND OUTPUT ---"
      echo
    } >> "$TASK_LOG"

    echo "[$(date '+%F %T')] 开始执行: $CONFIG_FILE" >> "$MASTER_LOG"
    python torch_main.py --cf "$CONFIG_FILE" >> "$TASK_LOG" 2>&1
    STATUS=$?

    {
      echo
      echo "--- TASK END: $(date) ---"
      echo "--- EXIT CODE: $STATUS ---"
    } >> "$TASK_LOG"

    if [ $STATUS -eq 0 ]; then
      echo "[$(date '+%F %T')] 完成: $CONFIG_FILE" >> "$MASTER_LOG"
    else
      echo "[$(date '+%F %T')] 失败: $CONFIG_FILE, exit code=$STATUS" >> "$MASTER_LOG"
      exit $STATUS
    fi
  )

  STATUS=$?
  if [ $STATUS -ne 0 ]; then
    echo "[$(date '+%F %T')] 因前一任务失败，停止后续任务" >> "$MASTER_LOG"
    exit $STATUS
  fi
done

echo "[$(date '+%F %T')] 全部任务执行完成" >> "$MASTER_LOG"
