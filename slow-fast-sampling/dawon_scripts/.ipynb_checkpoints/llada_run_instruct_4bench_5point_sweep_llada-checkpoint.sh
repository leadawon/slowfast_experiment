#!/bin/bash

set -euo pipefail

# Usage:
#   bash dawon_scripts/llada_run_instruct_4bench_5point_sweep_llada.sh [GPU_IDS] [MAIN_PORT]
# Example:
#   bash dawon_scripts/llada_run_instruct_4bench_5point_sweep_llada.sh 0 29510
GPU_IDS="${1:-0}"
MAIN_PORT="${2:-29510}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
GPU_IDS="$(echo "${GPU_IDS}" | tr -d '[:space:]')"
NUM_PROCESSES="$(echo "${GPU_IDS}" | awk -F',' '{print NF}')"

MODEL="GSAI-ML/LLaDA-8B-Instruct"
ACCEL_CONFIG="${PROJECT_ROOT}/eval_config.yaml"
DREAM_TASKS_PATH="${DREAM_TASKS_PATH:-/workspace/slowfast_experiment/data/tasks}"

# Shared sampling setup
TEMPERATURE="0.1"
TOP_P="0.9"
NUM_FEWSHOT="0"
LIMIT="${LIMIT:-9999}"
TASKS="${TASKS:-gsm8k_cot humaneval_instruct mbpp_instruct ifeval}"

OUTPUT_ROOT="${PROJECT_ROOT}/dawon_outputs/llada_instruct_4bench_5point_sweep_llada"

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

has_completed_result() {
  local task="$1"
  local output_path="$2"
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

task_enabled() {
    local task="$1"
    [[ " ${TASKS} " == *" ${task} "* ]]
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

# Conservative -> Aggressive points
POINT_LABELS=(
  "p1_conservative"
  "p2_safe"
  "p3_balanced"
  "p4_fast"
  "p5_aggressive"
)

GSM8K_STEPS=(256 128 64 32 16)
HUMANEVAL_STEPS=(512 256 128 64 32)
MBPP_STEPS=(768 384 192 96 48)
IFEVAL_STEPS=(768 384 192 96 48)

run_eval() {
    local task="$1"
    local max_new_tokens="$2"
    local block_length="$3"
    local point_label="$4"
    local output_path="${OUTPUT_ROOT}/${task}/${point_label}_step_${block_length}"

    echo "=================================================="
    echo "Starting evaluation for ${task} [${point_label}]"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, main_process_port=${MAIN_PORT}, num_processes=${NUM_PROCESSES}"
    echo "max_new_tokens=${max_new_tokens}, block_length=${block_length}, temperature=${TEMPERATURE}, top_p=${TOP_P}, fewshot=${NUM_FEWSHOT}"
    echo "limit=${LIMIT}"
    echo "output_path=${output_path}"

    if has_completed_result "${task}" "${output_path}"; then
        echo "Skip: completed result already exists under ${output_path}"
        return 0
    fi

    FORWARD_STATS_DIR="${output_path}" run_launcher --model LLADA \
        --model_args pretrained=${MODEL},parallelize=False,dtype=bfloat16,temperature=${TEMPERATURE},is_feature_cache=False,is_cfg_cache=False,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0 \
        --gen_kwargs gen_length=${max_new_tokens},block_length=${block_length},cfg_scale=0.0 \
        --tasks "${task}" \
        --num_fewshot "${NUM_FEWSHOT}" \
        --batch_size 1 \
        --limit "${LIMIT}" \
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
    shift 2
    local steps=("$@")

    local i
    for i in "${!POINT_LABELS[@]}"; do
        run_eval "${task}" "${max_new_tokens}" "${steps[$i]}" "${POINT_LABELS[$i]}"
    done
}

# 1) gsm8k_cot
if task_enabled "gsm8k_cot"; then
    run_task_sweep "gsm8k_cot" 256 "${GSM8K_STEPS[@]}"
fi

# 2) humaneval_instruct
if task_enabled "humaneval_instruct"; then
    run_task_sweep "humaneval_instruct" 512 "${HUMANEVAL_STEPS[@]}"
fi

# 3) mbpp_instruct
if task_enabled "mbpp_instruct"; then
    run_task_sweep "mbpp_instruct" 768 "${MBPP_STEPS[@]}"
fi

# 4) ifeval
if task_enabled "ifeval"; then
    run_task_sweep "ifeval" 768 "${IFEVAL_STEPS[@]}"
fi

echo "=================================================="
echo "All 5-point sweep evaluations completed successfully."
