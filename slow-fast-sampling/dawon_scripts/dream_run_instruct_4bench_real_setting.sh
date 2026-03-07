#!/bin/bash

set -euo pipefail

# Usage:
#   bash dawon_scripts/dream_run_instruct_4bench_real_setting.sh [GPU_IDS] [MAIN_PORT]
# Example:
#   bash dawon_scripts/dream_run_instruct_4bench_real_setting.sh 0 29510
GPU_IDS="${1:-0}"
MAIN_PORT="${2:-29510}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
GPU_IDS="$(echo "${GPU_IDS}" | tr -d '[:space:]')"
NUM_PROCESSES="$(echo "${GPU_IDS}" | awk -F',' '{print NF}')"

MODEL="Dream-org/Dream-v0-Instruct-7B"
ACCEL_CONFIG="${PROJECT_ROOT}/eval_config.yaml"
DREAM_TASKS_PATH="${DREAM_TASKS_PATH:-/workspace/Dream/eval_instruct/lm_eval/tasks}"

# Requested shared sampling setup
TEMPERATURE="0.1"
TOP_P="0.9"
NUM_FEWSHOT="0"

OUTPUT_ROOT="${PROJECT_ROOT}/dawon_outputs/dream_instruct_real_setting"

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

if [ ! -d "${DREAM_TASKS_PATH}" ]; then
    echo "ERROR: DREAM_TASKS_PATH not found: ${DREAM_TASKS_PATH}"
    echo "Please set DREAM_TASKS_PATH to Dream task registry path."
    exit 2
fi

run_launcher() {
    if [ "${NUM_PROCESSES}" -eq 1 ]; then
        python evaluation_script.py "$@"
    else
        accelerate launch --config_file "${ACCEL_CONFIG}" --main_process_port "${MAIN_PORT}" --gpu_ids "${GPU_IDS}" --num_processes "${NUM_PROCESSES}" evaluation_script.py "$@"
    fi
}

run_eval () {
    local task="$1"
    local max_new_tokens="$2"
    local diffusion_steps="$3"
    local output_path="${OUTPUT_ROOT}/${task}"

    echo "=================================================="
    echo "Starting evaluation for ${task}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, main_process_port=${MAIN_PORT}, num_processes=${NUM_PROCESSES}"
    echo "max_new_tokens=${max_new_tokens}, diffusion_steps=${diffusion_steps}, temperature=${TEMPERATURE}, top_p=${TOP_P}, fewshot=${NUM_FEWSHOT}"
    echo "output_path=${output_path}"

    FORWARD_STATS_DIR="${output_path}" run_launcher --model dream \
        --model_args pretrained=${MODEL},trust_remote_code=True,max_new_tokens=${max_new_tokens},diffusion_steps=${diffusion_steps},dtype=bfloat16,temperature=${TEMPERATURE},top_p=${TOP_P},alg=entropy,alg_temp=0.0,is_feature_cache=False,is_cfg_cache=False \
        --tasks "${task}" \
        --num_fewshot "${NUM_FEWSHOT}" \
        --batch_size 1 \
        --include_path "${DREAM_TASKS_PATH}" \
        --output_path "${output_path}" \
        --log_samples \
        --confirm_run_unsafe_code \
        --apply_chat_template

    echo "Completed evaluation for ${task}"
}

# 1) GSM8K
run_eval "gsm8k_cot" 256 256

# 2) HumanEval-Instruct
run_eval "humaneval_instruct" 512 512

# 3) MBPP-Instruct
run_eval "mbpp_instruct" 768 768

# 4) IFEval (Dream real_setting default)
run_eval "ifeval" 768 768

echo "=================================================="
echo "All evaluations completed successfully."
