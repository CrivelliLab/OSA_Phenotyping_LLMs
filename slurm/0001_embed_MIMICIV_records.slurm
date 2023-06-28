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
export MASTER_PORT=12340
export WORLD_SIZE=40

# The first hostname is the master address
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

#-
MODELS=( \
  "BioBART_base" \
  "BioBART_large" \
  "BioBERT_base" \
  "BioBERT_large" \
  "BioGPT_base" \
  "BioGPT_large" \
  "BioMegatron_base" \
  "Gatortron_base" \
  "Gatortron_s" \
  "Gatortron_medium" \
  "RadBERT_2m" \
  "RadBERT_4m" \
)
MAXBATCHES=( \
  12 \
  5 \
  36 \
  14 \
  6 \
  1 \
  10 \
  33 \
  33 \
  5 \
  36 \
  36 \
)
NBMODELS=${#MODELS[@]}

#- 
srun -n $WORLD_SIZE python src/1_embed_MIMICIV_records.py \
  --data /global/cfs/cdirs/m1532/Projects_MVP/_datasets/MIMIC_IV/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv \
  --model ${MODELS[$1]} \
  --max_batch ${MAXBATCHES[$1]} \
  --layer 0 \
  --out output/embeds/
