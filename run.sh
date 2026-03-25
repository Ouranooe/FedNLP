cat << 'EOF' > start_task.sh
#!/bin/bash

# 1. 定义文件路径
CONFIG_FILE="config/agnews/fedml_config_compare_llamaprox_24g.yaml"
LOG_DIR="log"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/moe_$(date +%Y%m%d_%H%M%S).log"

# 2. 预记录配置：把当前的 YAML 内容保存到日志头部，方便以后对比
echo "--- TASK START: $(date) ---" > $LOG_FILE
echo "--- CONFIG CONTENT ---" >> $LOG_FILE
cat $CONFIG_FILE >> $LOG_FILE
echo -e "\n--- COMMAND OUTPUT ---\n" >> $LOG_FILE

# 3. 后台运行并重定向输出
nohup python torch_main.py --cf $CONFIG_FILE >> $LOG_FILE 2>&1 &

echo "任务已在后台启动！"
echo "日志文件：$LOG_FILE"
echo "查看进度命令：tail -f $LOG_FILE"
EOF

# 授权并执行
chmod +x start_task.sh
./start_task.sh