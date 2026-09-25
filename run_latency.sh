#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/gahyun/miniconda3/envs/tf/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

WARMUP="${WARMUP:-1000}"
ITERS="${ITERS:-10000}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-logs/latency}"
mkdir -p "$LOG_DIR"

# Optional data overrides:
#   DATA_ROOT=/path/holding/exiD ./run_latency.sh
#   EXID_BASE_DIR=/path/to/exiD ./run_latency.sh
# Optional checkpoint overrides:
#   CKPT_ROOT=/path/to/grip_ckpts ./run_latency.sh
#   EXID_BASE_CKPT=/path/to/exiD0-5.pt EXID_I_CKPT=/path/to/exiD2-5.pt ./run_latency.sh

cases=(
  "exiD-baseline|configs/exiD0-5.yaml|ckpts/exiD0-5/best.pt"
  "exiD-+I|configs/exiD2-5.yaml|ckpts/exiD2-5/best.pt"
)

for row in "${cases[@]}"; do
  IFS='|' read -r name config ckpt <<< "$row"
  dataset="${name%%-*}"
  condition="${name#*-}"
  log_path="${LOG_DIR}/${name}.log"

  ckpt_key=""
  if [[ "$condition" == "baseline" ]]; then
    ckpt_key="${EXID_BASE_CKPT:-}"
  else
    ckpt_key="${EXID_I_CKPT:-}"
  fi
  if [[ -n "$ckpt_key" ]]; then
    ckpt="$ckpt_key"
  elif [[ -n "${CKPT_ROOT:-}" ]]; then
    ckpt="${CKPT_ROOT}/${ckpt#ckpts/}"
  fi

  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP] ${name}: missing ${ckpt}"
    continue
  fi

  cmd=(
    "$PYTHON_BIN" evaluate.py
    --config "$config"
    --ckpt "$ckpt"
    --measure_time
    --latency_warmup "$WARMUP"
    --latency_iters "$ITERS"
  )

  data_base_dir=""
  data_base_dir="${EXID_BASE_DIR:-}"
  if [[ -z "$data_base_dir" && -n "${DATA_ROOT:-}" ]]; then
    data_base_dir="${DATA_ROOT}/${dataset}"
  fi
  if [[ -n "$data_base_dir" ]]; then
    cmd+=(--data_base_dir "$data_base_dir")
  fi

  echo "[RUN] ${name}"
  printf '  %q' "${cmd[@]}"
  echo
  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi
  "${cmd[@]}" 2>&1 | tee "$log_path"
done
