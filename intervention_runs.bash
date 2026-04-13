#!/usr/bin/env bash
set -euo pipefail

# Run batch replication for all TI configurations across model families.
#
# Usage examples:
#   bash intervention_runs.bash
#   bash intervention_runs.bash --num_runs 5 --max_attempts 10 --cuda_visible_devices 0
#   bash intervention_runs.bash --enable_oom_retry_fallback -- --dataset_type all --num_shots 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

models=(
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "microsoft/Phi-3-mini-4k-instruct"
  "microsoft/Phi-3.5-mini-instruct"
  "google/gemma-2-2b-it"
  "google/gemma-7b-it"
  "meta-llama/Llama-3.2-1B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
)

ti_modes=(label_first structure_focus counterfactual suppress_cot)
fs_modes=(infer train)

NUM_RUNS=3
MAX_ATTEMPTS=5
CUDA_VISIBLE_DEVICES=0
SEED_STRATEGY="increment"
BASE_SEED=42
BATCH_OUTPUT_DIR="Results_TI/batch_runs"
ENABLE_OOM_RETRY_FALLBACK=false
OOM_RETRY_MAX_LEN=192
OOM_RETRY_PROMPT_BUDGET_TOKENS=192

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_runs)
      NUM_RUNS="$2"
      shift 2
      ;;
    --max_attempts)
      MAX_ATTEMPTS="$2"
      shift 2
      ;;
    --cuda_visible_devices)
      CUDA_VISIBLE_DEVICES="$2"
      shift 2
      ;;
    --seed_strategy)
      SEED_STRATEGY="$2"
      shift 2
      ;;
    --base_seed)
      BASE_SEED="$2"
      shift 2
      ;;
    --batch_output_dir)
      BATCH_OUTPUT_DIR="$2"
      shift 2
      ;;
    --enable_oom_retry_fallback)
      ENABLE_OOM_RETRY_FALLBACK=true
      shift
      ;;
    --no-enable_oom_retry_fallback)
      ENABLE_OOM_RETRY_FALLBACK=false
      shift
      ;;
    --oom_retry_max_len)
      OOM_RETRY_MAX_LEN="$2"
      shift 2
      ;;
    --oom_retry_prompt_budget_tokens)
      OOM_RETRY_PROMPT_BUDGET_TOKENS="$2"
      shift 2
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

OOM_RETRY_TOGGLE=("--no-enable_oom_retry_fallback")
if [[ "$ENABLE_OOM_RETRY_FALLBACK" == "true" ]]; then
  OOM_RETRY_TOGGLE=("--enable_oom_retry_fallback")
fi

echo "Starting TI batches for ${#models[@]} models"
echo "Batch settings: num_runs=${NUM_RUNS}, max_attempts=${MAX_ATTEMPTS}, cuda=${CUDA_VISIBLE_DEVICES}, seed_strategy=${SEED_STRATEGY}, base_seed=${BASE_SEED}, batch_output_dir=${BATCH_OUTPUT_DIR}, oom_retry_fallback=${ENABLE_OOM_RETRY_FALLBACK}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "Extra experiment args: ${EXTRA_ARGS[*]}"
fi

success_jobs=()
failed_jobs=()

for model in "${models[@]}"; do
  for fs in "${fs_modes[@]}"; do
    for ti in "${ti_modes[@]}"; do
      model_tag="${model//\//_}"
      batch_name="ti_${model_tag}_${fs}_${ti}_$(date +%Y%m%d-%H%M%S)"
      job="model=${model} fs=${fs} ti=${ti}"

      echo
      echo "===================================================================="
      echo "Running ${job}"
      echo "Batch name: ${batch_name}"

      if python3 run_intervention_batch.py \
        --num_runs "${NUM_RUNS}" \
        --max_attempts "${MAX_ATTEMPTS}" \
        --cuda_visible_devices "${CUDA_VISIBLE_DEVICES}" \
        --seed_strategy "${SEED_STRATEGY}" \
        --base_seed "${BASE_SEED}" \
        --batch_output_dir "${BATCH_OUTPUT_DIR}" \
        "${OOM_RETRY_TOGGLE[@]}" \
        --oom_retry_max_len "${OOM_RETRY_MAX_LEN}" \
        --oom_retry_prompt_budget_tokens "${OOM_RETRY_PROMPT_BUDGET_TOKENS}" \
        --batch_name "${batch_name}" \
        -- \
        --model_name "${model}" \
        --few_shot_mode "${fs}" \
        --ti_mode "${ti}" \
        "${EXTRA_ARGS[@]}"; then
        echo "Job completed successfully: ${job}"
        success_jobs+=("${job}")
      else
        echo "Job failed, continuing: ${job}"
        failed_jobs+=("${job}")
      fi
    done
  done
done

echo
echo "All TI batches completed."
echo "Successful jobs: ${#success_jobs[@]}"
for job in "${success_jobs[@]}"; do
  echo "  - ${job}"
done

echo "Failed jobs: ${#failed_jobs[@]}"
for job in "${failed_jobs[@]}"; do
  echo "  - ${job}"
done

if [[ ${#failed_jobs[@]} -gt 0 ]]; then
  exit 1
fi