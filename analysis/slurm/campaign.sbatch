#!/bin/bash
#SBATCH --job-name=spares-campaign
#SBATCH --partition=short
#SBATCH --array=0-13
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=03:55:00
#SBATCH --nice=50
#SBATCH --output=logs/%x_%A_%a.out

# cn001 hands out whole cores (two hardware threads), so each array task runs two
# single-threaded workers: 14 tasks = 28 logical CPUs, an eighth of the node.
# Workers stop cleanly before the 4 h limit; resubmitting picks up what is left.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spares
export TMPDIR=$HOME/tmp PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR"
cd "$HOME/passenger-ship-spares/analysis"
for k in 0 1; do
    python campaign.py --worker $((2 * SLURM_ARRAY_TASK_ID + k)) --nworkers 28 --hours 3.6 &
done
wait
