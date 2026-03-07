#!/bin/bash

set -euo pipefail

# Usage:
#   bash dawon_scripts/dream_run_instruct_4bench_5point_sweep_limited.sh [GPU_IDS] [MAIN_PORT]
# Example:
#   bash dawon_scripts/dream_run_instruct_4bench_5point_sweep_limited.sh 0 29510
GPU_IDS="${1:-3}"
MAIN_PORT="${2:-29510}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
GPU_IDS="$(echo "${GPU_IDS}" | tr -d '[:space:]')"
NUM_PROCESSES="$(echo "${GPU_IDS}" | awk -F',' '{print NF}')"

MODEL="Dream-org/Dream-v0-Instruct-7B"
ACCEL_CONFIG="${PROJECT_ROOT}/eval_config.yaml"
DREAM_TASKS_PATH="${DREAM_TASKS_PATH:-/workspace/Dream/eval_instruct/lm_eval/tasks}"

# Shared sampling setup
TEMPERATURE="0.1"
TOP_P="0.9"
NUM_FEWSHOT="0"

OUTPUT_ROOT="${PROJECT_ROOT}/dawon_outputs/dream_instruct_4bench_5point_sweep_limited"

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

run_launcher() {
    if [ "${NUM_PROCESSES}" -eq 1 ]; then
        python evaluation_script.py "$@"
    else
        accelerate launch --config_file "${ACCEL_CONFIG}" --main_process_port "${MAIN_PORT}" --gpu_ids "${GPU_IDS}" --num_processes "${NUM_PROCESSES}" evaluation_script.py "$@"
    fi
}

has_completed_result() {
    local task="$1"
    local output_path="$2"
    local stats_file
    stats_file="$(find "${output_path}" -type f -name 'forward_stats.json' -size +0c -print -quit 2>/dev/null || true)"
    if [ -z "${stats_file}" ]; then
        return 1
    fi

    local result_files
    local result_file
    result_files="$(find "${output_path}" -type f -name 'results_*.json' -size +0c 2>/dev/null || true)"
    while IFS= read -r result_file; do
        [ -z "${result_file}" ] && continue
        if result_matches_expected_task_config "${task}" "${result_file}"; then
            return 0
        fi
    done <<< "${result_files}"
    return 1
}

result_matches_expected_task_config() {
    local task="$1"
    local result_file="$2"

    python - "${task}" "${result_file}" <<'PY'
import json
import sys

task = sys.argv[1]
path = sys.argv[2]

with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)

versions = payload.get("versions") or {}
cfg = (payload.get("configs") or {}).get(task, {})
meta = cfg.get("metadata") or {}

ok = True
if task == "humaneval_instruct":
    ok = float(versions.get(task, -1)) == 2.0 and float(meta.get("version", -1)) == 2.0
elif task == "mbpp_instruct":
    # Dream task yaml signature: gen_prefix starts with "Here is ...", doc_to_target includes [DONE]
    gen_prefix = cfg.get("gen_prefix", "")
    doc_to_target = cfg.get("doc_to_target", "")
    ok = (
        float(versions.get(task, -1)) == 1.0
        and "Here is the completed function" in str(gen_prefix)
        and "[DONE]" in str(doc_to_target)
    )

sys.exit(0 if ok else 1)
PY
}

validate_task_path() {
    if [ ! -d "${DREAM_TASKS_PATH}" ]; then
        echo "ERROR: DREAM_TASKS_PATH not found: ${DREAM_TASKS_PATH}"
        echo "Please set DREAM_TASKS_PATH to Dream task registry path."
        return 1
    fi
    return 0
}

# Conservative -> Aggressive points
POINT_LABELS=(
  "p1_conservative"
  "p2_safe"
  "p3_balanced"
  "p4_fast"
  "p5_aggressive"
)

GSM8K_STEPS=(256 192 128 64 32)
HUMANEVAL_STEPS=(512 384 256 128 64)
MBPP_STEPS=(768 576 384 192 96)
IFEVAL_STEPS=(768 576 384 192 96)

run_eval() {
    local task="$1"
    local max_new_tokens="$2"
    local diffusion_steps="$3"
    local point_label="$4"
    local limit="$5"
    local output_path="${OUTPUT_ROOT}/${task}/${point_label}_step_${diffusion_steps}"

    echo "=================================================="
    echo "Starting evaluation for ${task} [${point_label}]"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, main_process_port=${MAIN_PORT}, num_processes=${NUM_PROCESSES}"
    echo "max_new_tokens=${max_new_tokens}, diffusion_steps=${diffusion_steps}, temperature=${TEMPERATURE}, top_p=${TOP_P}, fewshot=${NUM_FEWSHOT}, limit=${limit}"
    echo "output_path=${output_path}"

    if has_completed_result "${task}" "${output_path}"; then
        echo "Skip: completed result already exists under ${output_path}"
        return 0
    fi

    FORWARD_STATS_DIR="${output_path}" run_launcher --model dream \
        --model_args pretrained=${MODEL},trust_remote_code=True,max_new_tokens=${max_new_tokens},diffusion_steps=${diffusion_steps},dtype=bfloat16,temperature=${TEMPERATURE},top_p=${TOP_P},alg=entropy,alg_temp=0.0,is_feature_cache=False,is_cfg_cache=False \
        --tasks "${task}" \
        --num_fewshot "${NUM_FEWSHOT}" \
        --batch_size 1 \
        --limit "${limit}" \
        --include_path "${DREAM_TASKS_PATH}" \
        --output_path "${output_path}" \
        --log_samples \
        --confirm_run_unsafe_code \
        --apply_chat_template

    echo "Completed evaluation for ${task} [${point_label}]"
}

run_task_sweep() {
    local task="$1"
    local max_new_tokens="$2"
    local limit="$3"
    shift 3
    local steps=("$@")

    local i
    for i in "${!POINT_LABELS[@]}"; do
        run_eval "${task}" "${max_new_tokens}" "${steps[$i]}" "${POINT_LABELS[$i]}" "${limit}"
    done
}

# 1) gsm8k_cot (limit=200)
validate_task_path
run_task_sweep "gsm8k_cot" 256 200 "${GSM8K_STEPS[@]}"

# 2) humaneval_instruct (limit=30)
run_task_sweep "humaneval_instruct" 512 30 "${HUMANEVAL_STEPS[@]}"

# 3) mbpp_instruct (limit=20)
run_task_sweep "mbpp_instruct" 768 20 "${MBPP_STEPS[@]}"

# 4) ifeval (limit=50)
run_task_sweep "ifeval" 768 50 "${IFEVAL_STEPS[@]}"

echo "=================================================="
echo "All limited 5-point sweep evaluations completed successfully."
