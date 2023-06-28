#!/bin/bash
#SBATCH --account=m1532
#SBATCH --job-name=m1532.clustr_MIMICIV_records
#SBATCH --output=logs/m1532.clustr_MIMICIV_records.out
#SBATCH --error=logs/m1532.clustr_MIMICIV_records.err
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -n 1

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
#export MASTER_PORT=12335
#export WORLD_SIZE=4

# The first hostname is the master address
#master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#export MASTER_ADDR=$master_addr

#- 
python src/2_clustr_MIMICIV_reports.py \
  --embeds output/embeds/BioBART_base.discharge.layer.0/
