#!/bin/bash
#SBATCH --job-name=spares-diag
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# Same diagnosis as the local run, on the machine of the Table III run.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spares
export TMPDIR=$HOME/tmp PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR"
cd "$HOME/passenger-ship-spares/analysis"
python campaign_diag.py --k 10 --cases 0:29,1:4,1:6,1:9,1:19 --out ../data/v2/campaign_diag_hpc_a.json &
python campaign_diag.py --k 10 --cases 1:28,1:37,2:11,2:29,2:38 --out ../data/v2/campaign_diag_hpc_b.json &
wait
