#!/bin/bash
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 0:30:00
#SBATCH -n 32
#SBATCH --ntasks-per-node=8
#SBATCH -c 10
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=map_gpu:0,1,2,3,4,5,6,7,8

export SLURM_CPU_BIND="cores"
srun python NNsearch_NCBI_blueBERT_gpu.py \
	--df MIMICIII_TRIGRAMS.csv \
	--col trigrams \
	--search "obstructive sleep apnea" \
	--batchsize 32 \
	--outpath NNsearch_OSA/