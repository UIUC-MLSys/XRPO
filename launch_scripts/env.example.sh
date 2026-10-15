#!/bin/bash
# Copy to env.sh (git-ignored) and fill in. Then: source env.sh
#
#   cp launch_scripts/env.example.sh env.sh

# --- credentials -----------------------------------------------------------
export HUGGING_FACE_HUB_TOKEN=""      # https://huggingface.co/settings/tokens
export WANDB_API_KEY=""               # https://wandb.ai/authorize
export WANDB_ENTITY=""                # your wandb team/user
export WANDB_PROJECT="xrpo"

# --- paths -----------------------------------------------------------------
# Train/test parquet files produced by data/ (see README).
export DATA_DIR="./data"
# Merged ICL corpus + similar-question files (see icl_corpus/README.md).
export ICL_DIR="./icl_corpus"
# Checkpoints, validation dumps and the periodic ICL corpus snapshots.
export EXP_DIR="./experiments"

# --- hardware --------------------------------------------------------------
export GPU_NUM=4
