#!/bin/bash
#SBATCH --account=m1532
#SBATCH --job-name=m1532.biogpt.test
#SBATCH --output=logs/m1532.biogpt.test.out
#SBATCH --error=logs/m1532.biogpt.test.err
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -n 1
#SBATCH --gpus 4

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=4

# The first hostname is the master address
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

#--
srun -n 4 python src/biogpt.py