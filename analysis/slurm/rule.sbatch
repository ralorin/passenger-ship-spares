#!/bin/bash
#SBATCH --job-name=spares-rule
#SBATCH --partition=short
#SBATCH --array=0-10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:55:00
#SBATCH --output=logs/%x_%A_%a.out

# Bootstrap campaigns of Tables III and IV again with the convergence rule of
# hglmm.Fit.fit (SPARES_RULE=1): 11 tasks x 2 workers = 22 single-threaded workers.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spares
export TMPDIR=$HOME/tmp PYTHONUNBUFFERED=1 SPARES_RULE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR"
cd "$HOME/passenger-ship-spares/analysis"
for k in 0 1; do
    j=$((2 * SLURM_ARRAY_TASK_ID + k))
    (python campaign_reg.py --worker $j --nworkers 22 --hours 1.8; python campaign_boot.py --worker $j --nworkers 22 --hours 3.6) &
done
wait
