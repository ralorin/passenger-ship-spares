#!/bin/bash
#SBATCH --job-name=spares-reg
#SBATCH --partition=short
#SBATCH --array=0-6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:50:00
#SBATCH --output=logs/%x_%A_%a.out
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spares
export TMPDIR=$HOME/tmp PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR"
cd "$HOME/passenger-ship-spares/analysis"
for k in 0 1; do
    python campaign_reg.py --worker $((2 * SLURM_ARRAY_TASK_ID + k)) --nworkers 14 --hours 0.7 &
done
wait
