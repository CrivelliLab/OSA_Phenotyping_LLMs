#!/bin/bash
#SBATCH --account=m1532
#SBATCH --job-name=m1532.embed_MIMICIV_records
#SBATCH --output=logs/m1532.embed_MIMICIV_records.%j.out
#SBATCH --error=logs/m1532.embed_MIMICIV_records.%j.err
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -n 10
#SBATCH --gpus 40

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=$1
export WORLD_SIZE=40

# The first hostname is the master address
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

#- 
srun -n $WORLD_SIZE python src/1_embed_MIMICIV_records.py \
  --data /global/cfs/cdirs/m1532/Projects_MVP/_datasets/MIMIC_IV/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv \
  --model $2 \
  --max_batch $3 \
  --layer $4 \
  --out output/embeds/
