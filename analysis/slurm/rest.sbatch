#!/bin/bash
#SBATCH --job-name=spares-rest
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Finishes the queue of worker 2 of rule.sbatch on four more workers (files are written atomically).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spares
export TMPDIR=$HOME/tmp PYTHONUNBUFFERED=1 SPARES_RULE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$HOME/passenger-ship-spares/analysis"
for k in 0 1 2 3; do
  python -c "
import os, pandas as pd, campaign_boot as cb
done = set(os.listdir(cb.RAW))
todo = [(c, r) for k, (c, r) in enumerate(cb.tasks()) if k % 22 == 2 and \"c%03d_r%03d.csv\" % (c[\"cell\"], r) not in done][1:]
for c, r in todo[$k::4]:
    p = os.path.join(cb.RAW, \"c%03d_r%03d.csv\" % (c[\"cell\"], r))
    if os.path.exists(p): continue
    pd.DataFrame([cb.one_rep(c, r)]).to_csv(p + \".tmp2\", index=False); os.replace(p + \".tmp2\", p); print(c[\"cell\"], r, flush=True)
" &
done
wait
