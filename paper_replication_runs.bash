#!/usr/bin/env bash
set -euo pipefail

# Run replication batches across all models listed in the paper's results table.
#
# Usage examples:
#   bash paper_replication_runs.bash
#   bash paper_replication_runs.bash --dataset_type all --training_mode all
#   bash paper_replication_runs.bash --dataset_type climate --training_mode few-shot --few_shot_mode train
# 	GOOD OOM COMMAND BELOW:
# 	bash paper_replication_runs.bash --num_runs 5 --max_attempts 10 --cuda_visible_devices 0 --enable_oom_retry_fallback --oom_retry_max_len 192 --oom_retry_prompt_budget_tokens 192 -- --dataset_type all --training_mode few-shot --few_shot_mode train

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Model IDs matching the paper table model names.
MODELS=(
	"meta-llama/Llama-3.2-1B-Instruct" 
	"meta-llama/Llama-3.2-3B-Instruct"
	"google/gemma-7b-it"
	"google/gemma-2-2b-it"
	"microsoft/Phi-3-mini-4k-instruct"
	"microsoft/Phi-3.5-mini-instruct"
	"Qwen/Qwen2.5-3B-Instruct"
	"Qwen/Qwen2.5-1.5B-Instruct" 
)

# Fixed batch controls requested.
NUM_RUNS=5
MAX_ATTEMPTS=10
CUDA_VISIBLE_DEVICES=0
SEED_STRATEGY="increment"
BASE_SEED=42
ENABLE_OOM_RETRY_FALLBACK=false
OOM_RETRY_MAX_LEN=192
OOM_RETRY_PROMPT_BUDGET_TOKENS=192

# Optional passthrough args to paper_replication.py.
# Supported batch-level flags are consumed here; all others are forwarded.
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

echo "Starting replication batches for ${#MODELS[@]} models"
echo "Batch settings: num_runs=${NUM_RUNS}, max_attempts=${MAX_ATTEMPTS}, cuda=${CUDA_VISIBLE_DEVICES}, seed_strategy=${SEED_STRATEGY}, base_seed=${BASE_SEED}, oom_retry_fallback=${ENABLE_OOM_RETRY_FALLBACK}, oom_retry_max_len=${OOM_RETRY_MAX_LEN}, oom_retry_prompt_budget_tokens=${OOM_RETRY_PROMPT_BUDGET_TOKENS}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
	echo "Extra experiment args: ${EXTRA_ARGS[*]}"
fi

SUCCESS_MODELS=()
FAILED_MODELS=()

for model in "${MODELS[@]}"; do
	model_tag="${model//\//_}"
	batch_name="paper_${model_tag}_$(date +%Y%m%d-%H%M%S)"

	echo
	echo "===================================================================="
	echo "Running model: ${model}"
	echo "Batch name: ${batch_name}"

	if python run_replication_batch.py \
		--num_runs "${NUM_RUNS}" \
		--max_attempts "${MAX_ATTEMPTS}" \
		--cuda_visible_devices "${CUDA_VISIBLE_DEVICES}" \
		--seed_strategy "${SEED_STRATEGY}" \
		--base_seed "${BASE_SEED}" \
		"${OOM_RETRY_TOGGLE[@]}" \
		--oom_retry_max_len "${OOM_RETRY_MAX_LEN}" \
		--oom_retry_prompt_budget_tokens "${OOM_RETRY_PROMPT_BUDGET_TOKENS}" \
		--batch_name "${batch_name}" \
		-- \
		--model_name "${model}" \
		"${EXTRA_ARGS[@]}"; then
		echo "Model completed successfully: ${model}"
		SUCCESS_MODELS+=("${model}")
	else
		echo "Model failed, continuing to next model: ${model}"
		FAILED_MODELS+=("${model}")
	fi
done

echo
echo "All model batches completed."
echo "Successful models: ${#SUCCESS_MODELS[@]}"
for model in "${SUCCESS_MODELS[@]}"; do
	echo "  - ${model}"
done

echo "Failed models: ${#FAILED_MODELS[@]}"
for model in "${FAILED_MODELS[@]}"; do
	echo "  - ${model}"
done

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
	exit 1
fi
