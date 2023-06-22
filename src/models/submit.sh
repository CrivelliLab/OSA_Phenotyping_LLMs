#- Test these LLMs
mkdir logs/
rm logs/*
sbatch slurm/biogpt.slurm
sbatch slurm/biobart.slurm
sbatch slurm/biobert.slurm
sbatch slurm/biomegatron.slurm
#sbatch slurm/clinicalt5.slurm
sbatch slurm/gatortron.slurm
sbatch slurm/radbert.slurm