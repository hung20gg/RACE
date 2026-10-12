#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

model_name="${MODEL_NAME:-gpt-oss-20b}"
dataset_name="${DATASET_NAME:-NQ}"
model_path="${MODEL_PATH:-openai/gpt-oss-20b}"
gpu_id="${GPU_ID:-0}"
output_dir="${OUTPUT_DIR:-./modeloutput}"
args=(--model_name "$model_name" --model_path "$model_path"
      --dataset_name "$dataset_name" --gpu_id "$gpu_id" --output_dir "$output_dir"
      --reasoning_effort "${REASONING_EFFORT:-medium}" --current_date "${CURRENT_DATE:-2026-10-12}"
      --max_new_tokens "${MAX_NEW_TOKENS:-2048}" --num_samples "${NUM_SAMPLES:-5}")
if [[ -n "${MAX_EXAMPLES:-}" ]]; then args+=(--max_examples "$MAX_EXAMPLES"); fi
if [[ "${LOCAL_FILES_ONLY:-0}" == 1 ]]; then args+=(--local_files_only); fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    CUDA_VISIBLE_DEVICES="" python generate.py "${args[@]}" --dry_run
    exit 0
fi
python generate.py "${args[@]}"

python cot_extraction.py \
    --cot_extractor_model_path bebr2/RACE-CoT-Extractor-Llama-8B \
    --model_name "$model_name" --dataset_name "$dataset_name" \
    --data_path "$output_dir" --output_dir "$output_dir" --gpu_ids "$gpu_id"

CUDA_VISIBLE_DEVICES="$gpu_id" python main.py \
    --dataset "$dataset_name" --model "$model_name" --data_dir "$output_dir" \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --nli_model potsawee/deberta-v3-large-mnli \
    --llm_model bebr2/RACE-CoT-Extractor-Llama-8B \
    --sindex_threshold 0.9 --output "./${model_name}_race_score.json" --gpu

if [[ -f "$output_dir/$dataset_name/$model_name/judge.json" && "$output_dir" == ./modeloutput ]]; then
    python eval.py "$model_name" "$dataset_name"
else
    echo "Scores saved. AUROC requires aligned judge.json labels in ./modeloutput."
fi
