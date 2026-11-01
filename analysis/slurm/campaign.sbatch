#!/bin/bash
#SBATCH --job-name=spares-campaign
#SBATCH --partition=short
#SBATCH --array=0-27
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=03:55:00
#SBATCH --nice=50
#SBATCH --output=logs/%x_%A_%a.out

# 28 single-core workers (an eighth of cn001). Each one stops cleanly before the
# 4 h limit; resubmitting the same script picks up the unfinished blocks.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spares
export TMPDIR=$HOME/tmp PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR"
cd "$HOME/passenger-ship-spares/analysis"
python campaign.py --worker "$SLURM_ARRAY_TASK_ID" --nworkers 28 --hours 3.6
