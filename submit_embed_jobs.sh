#- LLM MIMIC-IV Embedding Jobs: Last Layer
#sbatch slurm/0001_embed_MIMICIV_records.slurm 12320 "BioBART_base" 12 0
#sbatch slurm/0001_embed_MIMICIV_records.slurm 12321 "BioBART_large" 5 0
#sbatch slurm/0001_embed_MIMICIV_records.slurm 12322 "BioBERT_base" 36 0
#sbatch slurm/0001_embed_MIMICIV_records.slurm 12323 "BioBERT_large" 14 0
#sbatch slurm/0001_embed_MIMICIV_records.slurm 12324 "BioGPT_base" 6 0
#sbatch slurm/0001_embed_MIMICIV_records.slurm 12325 "BioGPT_large" 1 0
#sbatch slurm/0001_embed_MIMICIV_records.slurm 12326 "BioMegatron_base" 10  0
sbatch slurm/0001_embed_MIMICIV_records.slurm 12327 "Gatortron_base" 33 0
#sbatch slurm/0001_embed_MIMICIV_records.slurm 12328 "Gatortron_s" 33 0
#sbatch slurm/0001_embed_MIMICIV_records.slurm 12329 "Gatortron_medium" 5 0
sbatch slurm/0001_embed_MIMICIV_records.slurm 12330 "RadBERT_2m" 36 0
#sbatch slurm/0001_embed_MIMICIV_records.slurm 12331 "RadBERT_4m" 36 0
