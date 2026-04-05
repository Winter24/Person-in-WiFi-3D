#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/work_dirs/paper_launch_logs}"
DRY_RUN="${DRY_RUN:-0}"

RUN_IDS=(B0 B1 B2 B4 B5)
RUN_GPU_IDS=()
PIDS=()
FAILURES=0

declare -A CONFIG_PATHS=(
  [B0]="configs/wifi/petr_wifi.py"
  [B1]="configs/wifi/petr_wifi.py"
  [B2]="configs/wifi/petr_wifi_mamba.py"
  [B4]="configs/wifi/wi_tidir_wifi.py"
  [B5]="configs/wifi/wi_tidir_wifi.py"
)

declare -A CFG_OPTIONS=(
  [B0]=""
  [B1]="model.backbone.mode=spectral"
  [B2]=""
  [B4]="model.bbox_head.loss_bone=None"
  [B5]=""
)

declare -A RUN_STATUS=()
declare -A RUN_LOG_PATH=()
declare -A RUN_PID=()

mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCHER_LOG="$LOG_DIR/launcher_${TIMESTAMP}.log"
touch "$LAUNCHER_LOG"
exec > >(tee -a "$LAUNCHER_LOG") 2>&1

print_header() {
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: Required command not found: $cmd" >&2
    exit 1
  fi
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s\n' "$value"
}

validate_repo_files() {
  local run_id config_path
  for run_id in "${RUN_IDS[@]}"; do
    config_path="$ROOT_DIR/${CONFIG_PATHS[$run_id]}"
    if [[ ! -f "$config_path" ]]; then
      echo "ERROR: Missing config for $run_id: $config_path" >&2
      exit 1
    fi
  done
  if [[ ! -f "$ROOT_DIR/tools/train.py" ]]; then
    echo "ERROR: Missing train entrypoint: $ROOT_DIR/tools/train.py" >&2
    exit 1
  fi
}

discover_gpus() {
  local -n out_indices_ref="$1"
  local -n out_rows_ref="$2"
  local line index name mem_total mem_used util

  mapfile -t out_rows_ref < <(
    "$NVIDIA_SMI_BIN" \
      --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )

  if [[ "${#out_rows_ref[@]}" -lt 5 ]]; then
    echo "ERROR: Need at least 5 visible GPUs, found ${#out_rows_ref[@]}." >&2
    exit 1
  fi

  out_indices_ref=()
  for line in "${out_rows_ref[@]}"; do
    IFS=',' read -r index name mem_total mem_used util <<<"$line"
    index="$(trim "$index")"
    out_indices_ref+=("$index")
  done
}

validate_gpu_ids() {
  local -n visible_indices_ref="$1"
  local -a requested_ids=("$@")
  requested_ids=("${requested_ids[@]:1}")
  local requested visible found
  declare -A seen=()

  if [[ "${#requested_ids[@]}" -ne 5 ]]; then
    echo "ERROR: GPU_IDS must contain exactly 5 GPU IDs. Got: ${#requested_ids[@]}" >&2
    exit 1
  fi

  for requested in "${requested_ids[@]}"; do
    if [[ -n "${seen[$requested]:-}" ]]; then
      echo "ERROR: GPU_IDS contains duplicate GPU id: $requested" >&2
      exit 1
    fi
    seen[$requested]=1

    found=0
    for visible in "${visible_indices_ref[@]}"; do
      if [[ "$requested" == "$visible" ]]; then
        found=1
        break
      fi
    done
    if [[ "$found" -ne 1 ]]; then
      echo "ERROR: Requested GPU id $requested is not visible in nvidia-smi output." >&2
      exit 1
    fi
  done
}

select_gpu_ids() {
  local -n visible_indices_ref="$1"
  if [[ -n "${GPU_IDS:-}" ]]; then
    read -r -a RUN_GPU_IDS <<<"$GPU_IDS"
    validate_gpu_ids visible_indices_ref "${RUN_GPU_IDS[@]}"
  else
    RUN_GPU_IDS=("${visible_indices_ref[@]:0:5}")
  fi
}

print_gpu_summary() {
  local -n gpu_rows_ref="$1"
  local run_id gpu_id i

  print_header "Visible GPUs"
  printf "%-8s %-28s %-12s %-12s %-10s\n" "GPU_ID" "NAME" "MEM_TOTAL" "MEM_USED" "UTIL(%)"
  for row in "${gpu_rows_ref[@]}"; do
    IFS=',' read -r gpu_id name mem_total mem_used util <<<"$row"
    printf "%-8s %-28s %-12s %-12s %-10s\n" \
      "$(trim "$gpu_id")" \
      "$(trim "$name")" \
      "$(trim "$mem_total") MiB" \
      "$(trim "$mem_used") MiB" \
      "$(trim "$util")"
  done

  print_header "Selected GPU Mapping"
  echo "Selected GPU IDs: ${RUN_GPU_IDS[*]}"
  for i in "${!RUN_IDS[@]}"; do
    run_id="${RUN_IDS[$i]}"
    gpu_id="${RUN_GPU_IDS[$i]}"
    echo "$run_id -> GPU $gpu_id"
  done
}

build_command() {
  local run_id="$1"
  local gpu_id="$2"
  local run_mode="$3"
  local work_dir="$ROOT_DIR/work_dirs/paper/$run_id"
  local config_path="$ROOT_DIR/${CONFIG_PATHS[$run_id]}"
  local cfg_option="${CFG_OPTIONS[$run_id]}"
  local cmd=(
    env
    "CUDA_VISIBLE_DEVICES=$gpu_id"
    "PYTHONHASHSEED=42"
    "CUBLAS_WORKSPACE_CONFIG=:4096:8"
    "$PYTHON_BIN"
    "$ROOT_DIR/tools/train.py"
    "$config_path"
    --gpu-id 0
    --seed 42
    --deterministic
    --work-dir "$work_dir"
  )

  if [[ "$run_mode" == "auto-resume" ]]; then
    cmd+=(--auto-resume)
  fi

  if [[ -n "$cfg_option" ]]; then
    cmd+=(--cfg-options "$cfg_option")
  fi

  printf '%q ' "${cmd[@]}"
  printf '\n'
}

terminate_children() {
  local pid
  echo
  echo "Received termination signal. Stopping child processes..."
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
  wait || true
  exit 130
}

launch_run() {
  local run_id="$1"
  local gpu_id="$2"
  local work_dir="$ROOT_DIR/work_dirs/paper/$run_id"
  local log_path="$LOG_DIR/${run_id}.log"
  local pid_path="$LOG_DIR/${run_id}.pid"
  local cmd
  local run_mode="fresh"

  mkdir -p "$work_dir"
  RUN_LOG_PATH[$run_id]="$log_path"
  if [[ -f "$work_dir/latest.pth" ]]; then
    run_mode="auto-resume"
  fi
  RUN_STATUS[$run_id]="$run_mode"
  cmd="$(build_command "$run_id" "$gpu_id" "$run_mode")"

  echo "[$run_id] mode=$run_mode gpu=$gpu_id"
  echo "[$run_id] log=$log_path"
  echo "[$run_id] cmd=$cmd"

  if [[ "$DRY_RUN" == "1" ]]; then
    RUN_PID[$run_id]="dry-run"
    return 0
  fi

  eval "$cmd" >"$log_path" 2>&1 &
  RUN_PID[$run_id]="$!"
  PIDS+=("$!")
  printf '%s\n' "$!" >"$pid_path"
}

wait_for_runs() {
  local run_id pid exit_code

  for run_id in "${RUN_IDS[@]}"; do
    pid="${RUN_PID[$run_id]}"
    if [[ "$pid" == "dry-run" ]]; then
      RUN_STATUS[$run_id]="dry-run"
      continue
    fi

    if wait "$pid"; then
      exit_code=0
      RUN_STATUS[$run_id]="ok"
    else
      exit_code=$?
      RUN_STATUS[$run_id]="failed($exit_code)"
      FAILURES=$((FAILURES + 1))
    fi
  done
}

print_final_summary() {
  local run_id gpu_id pid status log_path i

  print_header "Final Run Summary"
  printf "%-6s %-8s %-12s %-16s %s\n" "RUN" "GPU" "PID" "STATUS" "LOG"
  for i in "${!RUN_IDS[@]}"; do
    run_id="${RUN_IDS[$i]}"
    gpu_id="${RUN_GPU_IDS[$i]}"
    pid="${RUN_PID[$run_id]:-n/a}"
    status="${RUN_STATUS[$run_id]:-unknown}"
    log_path="${RUN_LOG_PATH[$run_id]:-n/a}"
    printf "%-6s %-8s %-12s %-16s %s\n" "$run_id" "$gpu_id" "$pid" "$status" "$log_path"
  done
  echo "Launcher log: $LAUNCHER_LOG"
}

main() {
  local visible_gpu_ids=()
  local visible_gpu_rows=()
  local i

  cd "$ROOT_DIR"

  require_command "$PYTHON_BIN"
  require_command "$NVIDIA_SMI_BIN"
  validate_repo_files
  discover_gpus visible_gpu_ids visible_gpu_rows
  select_gpu_ids visible_gpu_ids
  print_gpu_summary visible_gpu_rows

  if [[ "$DRY_RUN" == "1" ]]; then
    print_header "Dry Run"
    echo "DRY_RUN=1, commands will be printed but not executed."
  else
    trap terminate_children INT TERM
  fi

  for i in "${!RUN_IDS[@]}"; do
    launch_run "${RUN_IDS[$i]}" "${RUN_GPU_IDS[$i]}"
  done

  if [[ "$DRY_RUN" == "1" ]]; then
    print_final_summary
    return 0
  fi

  wait_for_runs
  print_final_summary

  if [[ "$FAILURES" -gt 0 ]]; then
    echo "ERROR: $FAILURES run(s) failed." >&2
    exit 1
  fi

  echo "All runs completed successfully."
}

main "$@"
