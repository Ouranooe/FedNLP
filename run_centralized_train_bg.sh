#!/usr/bin/env bash
set -euo pipefail

# Background launcher for centralized (non-federated) training.
# It records runtime metadata, full command line, and config snapshot into one log file.

CONFIG="config/simulation/fedml_config_compare_llama_24g copy.yaml"
MODEL_TYPE=""
EPOCHS="30"
MODEL_NAME=""
BATCH_SIZE=""
EVAL_BATCH_SIZE=""
MAX_SEQ_LENGTH=""
LEARNING_RATE=""
OUTPUT_DIR=""
MAX_TRAIN_SAMPLES=""
RUN_NAME=""
ENABLE_FP16=0
HEAD_ONLY=0

EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash run_centralized_train_bg.sh [options] [-- extra train.py args]

Options:
  --config PATH            YAML config path (default: config/simulation/fedml_config_compare_llama_24g.yaml)
  --model_type TYPE        llama | mor_llama
  --epochs N               Training epochs
  --model_name NAME        HuggingFace model name/path
  --batch_size N           Train batch size
  --eval_batch_size N      Eval batch size
  --max_seq_length N       Max sequence length
  --learning_rate LR       Learning rate
  --output_dir DIR         Output directory for checkpoints
  --max_train_samples N    Cap max number of training samples
  --run_name NAME          Optional run name prefix for logs
  --fp16                   Enable fp16
  --head_only              Freeze all params except classifier head
  -h, --help               Show this help

Examples:
  bash run_centralized_train_bg.sh \
    --config config/simulation/fedml_config_compare_llama_24g.yaml \
    --model_type llama --epochs 3 --head_only --output_dir ./outputs/centralized_llama

  bash run_centralized_train_bg.sh \
    --config config/simulation/fedml_config_compare_mor_expert_24g.yaml \
    --model_type mor_llama --epochs 3 --head_only --output_dir ./outputs/centralized_mor
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"; shift 2 ;;
    --model_type)
      MODEL_TYPE="$2"; shift 2 ;;
    --epochs)
      EPOCHS="$2"; shift 2 ;;
    --model_name)
      MODEL_NAME="$2"; shift 2 ;;
    --batch_size)
      BATCH_SIZE="$2"; shift 2 ;;
    --eval_batch_size)
      EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --max_seq_length)
      MAX_SEQ_LENGTH="$2"; shift 2 ;;
    --learning_rate)
      LEARNING_RATE="$2"; shift 2 ;;
    --output_dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --max_train_samples)
      MAX_TRAIN_SAMPLES="$2"; shift 2 ;;
    --run_name)
      RUN_NAME="$2"; shift 2 ;;
    --fp16)
      ENABLE_FP16=1; shift ;;
    --head_only)
      HEAD_ONLY=1; shift ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1 ;;
  esac
done

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file not found: $CONFIG"
  exit 1
fi

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
NAME_PART="${RUN_NAME:-centralized_train}"
LOG_FILE="${LOG_DIR}/${NAME_PART}_${TS}.log"
PID_FILE="${LOG_DIR}/${NAME_PART}_${TS}.pid"

CMD=(python train.py --config "$CONFIG")

if [[ -n "$MODEL_TYPE" ]]; then CMD+=(--model_type "$MODEL_TYPE"); fi
if [[ -n "$EPOCHS" ]]; then CMD+=(--epochs "$EPOCHS"); fi
if [[ -n "$MODEL_NAME" ]]; then CMD+=(--model_name "$MODEL_NAME"); fi
if [[ -n "$BATCH_SIZE" ]]; then CMD+=(--batch_size "$BATCH_SIZE"); fi
if [[ -n "$EVAL_BATCH_SIZE" ]]; then CMD+=(--eval_batch_size "$EVAL_BATCH_SIZE"); fi
if [[ -n "$MAX_SEQ_LENGTH" ]]; then CMD+=(--max_seq_length "$MAX_SEQ_LENGTH"); fi
if [[ -n "$LEARNING_RATE" ]]; then CMD+=(--learning_rate "$LEARNING_RATE"); fi
if [[ -n "$OUTPUT_DIR" ]]; then CMD+=(--output_dir "$OUTPUT_DIR"); fi
if [[ -n "$MAX_TRAIN_SAMPLES" ]]; then CMD+=(--max_train_samples "$MAX_TRAIN_SAMPLES"); fi
if [[ "$ENABLE_FP16" -eq 1 ]]; then CMD+=(--fp16); fi
if [[ "$HEAD_ONLY" -eq 1 ]]; then CMD+=(--head_only); fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then CMD+=("${EXTRA_ARGS[@]}"); fi

CMD_STR="$(printf '%q ' "${CMD[@]}")"

{
  echo "========================================"
  echo "Centralized Training Background Launch"
  echo "start_time: $(date '+%F %T %Z')"
  echo "host: $(hostname)"
  echo "user: ${USER:-unknown}"
  echo "cwd: $(pwd)"
  echo "python: $(command -v python || true)"
  echo "python_version: $(python --version 2>&1 || true)"
  echo "git_head: $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
  echo "git_branch: $(git branch --show-current 2>/dev/null || echo 'N/A')"
  echo "git_status_short:"
  git status --short 2>/dev/null || true
  echo "----------------------------------------"
  echo "config_path: $CONFIG"
  echo "command: $CMD_STR"
  echo "----------------------------------------"
  echo "config_snapshot_begin"
  sed -n '1,260p' "$CONFIG"
  echo "config_snapshot_end"
  echo "----------------------------------------"
} > "$LOG_FILE"

nohup bash -lc "$CMD_STR" >> "$LOG_FILE" 2>&1 &
PID=$!

echo "$PID" > "$PID_FILE"

echo "Started in background."
echo "PID: $PID"
echo "PID file: $PID_FILE"
echo "Log file: $LOG_FILE"
echo "Track log: tail -f $LOG_FILE"
echo "Stop run: kill $PID"
