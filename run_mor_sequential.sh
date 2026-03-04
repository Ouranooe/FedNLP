#!/bin/bash

mkdir -p logs

YAML_FILES=(
    "config/simulation/fedml_config_mor_1.yaml"
    "config/simulation/fedml_config_mor_2.yaml"
    "config/simulation/fedml_config_mor_3.yaml"
)

for i in "${!YAML_FILES[@]}"; do
    YAML=${YAML_FILES[$i]}
    IDX=$((i + 1))
    LOG=logs/run_mor_${IDX}_$(date +%Y%m%d_%H%M%S).log
    
    echo "========== RUN START ==========" > $LOG
    echo "Time: $(date)" >> $LOG
    echo "Host: $(hostname)" >> $LOG
    echo "CUDA_VISIBLE_DEVICES=3" >> $LOG
    echo "---------- YAML CONFIG ----------" >> $LOG
    cat $YAML >> $LOG
    echo "---------------------------------" >> $LOG
    echo "" >> $LOG
    
    CUDA_VISIBLE_DEVICES=3 nohup python torch_main.py \
        --cf $YAML \
        >> $LOG 2>&1 &
    
    echo "[$IDX/${#YAML_FILES[@]}] Started: $YAML (PID: $!)"
    
    # 如果不是最后一个，等待2.5小时
    nohup ./run_mor_sequential.sh > logs/scheduler.log 2>&1 &
    fi
done

echo "All jobs launched."
