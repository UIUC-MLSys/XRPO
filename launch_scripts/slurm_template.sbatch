#!/bin/bash
# Slurm wrapper template. Fill in the site-specific values (account, partition,
# module names, conda path) for your own cluster, then:
#
#   sbatch launch_scripts/slurm_template.sbatch launch_scripts/<run_script>.sh
#
#SBATCH --job-name=xrpo
#SBATCH --time=45:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
### #SBATCH --account=<your_allocation>
### #SBATCH --partition=<your_gpu_partition>

set -euo pipefail

### site-specific setup, e.g.:
### module load cuda/12.4.0
### source "$HOME/miniconda3/etc/profile.d/conda.sh"
### conda activate verl

unset ROCR_VISIBLE_DEVICES      # only needed on some AMD/ROCm-aware sites

source ./env.sh                 # credentials + DATA_DIR / ICL_DIR / EXP_DIR

bash "$1"
