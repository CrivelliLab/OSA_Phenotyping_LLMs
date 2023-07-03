#!/bin/bash
#SBATCH --account=m1532
#SBATCH --job-name=m1532.umap_MIMICIV_records
#SBATCH --output=logs/m1532.umap_MIMICIV_records.%j.out
#SBATCH --error=logs/m1532.umap_MIMICIV_records.%j.err
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -n 1

#- 
python src/3_umap_MIMICIV_reports.py \
  --embeds $1 \
  --nclusters $2 \
  --nneighbors $3 \
  --mindist $4 
