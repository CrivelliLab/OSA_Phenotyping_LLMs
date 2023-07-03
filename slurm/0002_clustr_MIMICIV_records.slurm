#!/bin/bash
#SBATCH --account=m1532
#SBATCH --job-name=m1532.clustr_MIMICIV_records
#SBATCH --output=logs/m1532.clustr_MIMICIV_records.%j.out
#SBATCH --error=logs/m1532.clustr_MIMICIV_records.%j.err
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH -n 1

#- 
python src/2_clustr_MIMICIV_reports.py \
  --embeds $1 \
  --nclusters $2
