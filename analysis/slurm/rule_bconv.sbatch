#!/bin/bash
#SBATCH --job-name=spares-rbconv
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:55:00
#SBATCH --output=logs/%x_%j.out
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spares
export TMPDIR=$HOME/tmp PYTHONUNBUFFERED=1 SPARES_RULE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR"
cd "$HOME/passenger-ship-spares/analysis"
python campaign_bconv.py --workers 4
