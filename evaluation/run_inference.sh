#!/bin/bash
# Batched vLLM inference over the evaluation sets.
#
#   bash evaluation/run_inference.sh
#
# Override any of the variables below from the environment, e.g.
#   MODEL=./experiments/my_run/global_step_120/actor bash evaluation/run_inference.sh
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/inference_results}"

# Reported settings: 16K response budget for Qwen3-1.7B, 8K for Qwen2.5-7B-Instruct.
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/${MAX_TOKENS}}"

NUM_GENERATIONS="${NUM_GENERATIONS:-32}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"

mkdir -p "$OUTPUT_DIR"

# --- math ------------------------------------------------------------------
python evaluation/run_inference.py \
    --model "$MODEL" \
    --input-parquets \
        "$DATA_DIR/test_aime2024/test.parquet" \
        "$DATA_DIR/test_aime2025/test.parquet" \
        "$DATA_DIR/test_amc/test.parquet" \
        "$DATA_DIR/test_brumo2025/test.parquet" \
        "$DATA_DIR/test_hmmt_feb_2025/test.parquet" \
    --output-dir "$OUTPUT_DIR" \
    --num-generations "$NUM_GENERATIONS" \
    --max-tokens "$MAX_TOKENS" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --temperature 1.0 \
    --top-p 0.95 \
    --gpu-memory-utilization 0.8 \
    --enable-thinking

# --- code ------------------------------------------------------------------
python evaluation/run_inference.py \
    --model "$MODEL" \
    --input-parquets \
        "$DATA_DIR/test_codeforces.parquet" \
        "$DATA_DIR/test_lcbv5.parquet" \
    --output-dir "$OUTPUT_DIR" \
    --num-generations "$NUM_GENERATIONS" \
    --max-tokens "$MAX_TOKENS" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --temperature 1.0 \
    --top-p 0.95 \
    --gpu-memory-utilization 0.8 \
    --enable-thinking

echo "Inference completed. Results saved to: $OUTPUT_DIR"
