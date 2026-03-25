cat << 'EOF' > start_task.sh
#!/bin/bash

TARGET_PID=3935
CHECK_INTERVAL=10

CONFIG_FILE="config/agnews/fedml_config_compare_mor_expert_adaptive_warmup_24g.yaml"
LOG_DIR="log"
mkdir -p "$LOG_DIR"

WRAPPER_LOG="$LOG_DIR/wait_and_start_$(date +%Y%m%d_%H%M%S).log"
LOG_FILE="$LOG_DIR/moe_$(date +%Y%m%d_%H%M%S).log"

# 如果不是后台模式，就把整个脚本自己丢到后台
if [ "$1" != "--daemon" ]; then
    nohup bash "$0" --daemon > "$WRAPPER_LOG" 2>&1 &
    BG_PID=$!
    echo "等待脚本已在后台启动！"
    echo "后台PID：$BG_PID"
    echo "等待日志：$WRAPPER_LOG"
    echo "训练日志：$LOG_FILE"
    exit 0
fi

echo "[$(date '+%F %T')] 等待进程 ${TARGET_PID} 结束后再启动任务..."
echo "[$(date '+%F %T')] 训练日志文件：$LOG_FILE"

# 先等待目标 PID 消失
while kill -0 "$TARGET_PID" 2>/dev/null; do
    echo "[$(date '+%F %T')] PID ${TARGET_PID} 仍在运行，等待 ${CHECK_INTERVAL}s ..."
    sleep "$CHECK_INTERVAL"
done

echo "[$(date '+%F %T')] 检测到 PID ${TARGET_PID} 已结束，开始启动任务。"

# 预记录配置
{
    echo "--- TASK START: $(date) ---"
    echo "--- CONFIG CONTENT ---"
    cat "$CONFIG_FILE"
    echo
    echo "--- COMMAND OUTPUT ---"
    echo
} >> "$LOG_FILE"

# 后台运行训练
nohup python torch_main.py --cf "$CONFIG_FILE" >> "$LOG_FILE" 2>&1 &

NEW_PID=$!

echo "[$(date '+%F %T')] 任务已在后台启动！"
echo "[$(date '+%F %T')] 新任务 PID：$NEW_PID"
echo "[$(date '+%F %T')] 训练日志：$LOG_FILE"
EOF

chmod +x start_task.sh
./start_task.sh