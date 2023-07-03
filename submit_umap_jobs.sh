#-
MODELS=( \
  #"output/embeds/BioBART_base.discharge.layer.0/" \
  #"output/embeds/BioBART_large.discharge.layer.0/" \
  #"output/embeds/BioBERT_base.discharge.layer.0/" \
  #"output/embeds/BioBERT_large.discharge.layer.0/" \
  #"output/embeds/BioGPT_base.discharge.layer.0/" \
  #"output/embeds/BioGPT_large.discharge.layer.0/" \
  #"output/embeds/BioMegatron_base.discharge.layer.0/" \
  #"output/embeds/Gatortron_base.discharge.layer.0/" \
  #"output/embeds/Gatortron_s.discharge.layer.0/" \
  #"output/embeds/Gatortron_medium.discharge.layer.0/" \
  #"output/embeds/RadBERT_2m.discharge.layer.0/" \
  #"output/embeds/RadBERT_4m.discharge.layer.0/" \
)

#- Umap Dimensionality Reduction Jobs
for i in {0..0} 
do 
  sbatch slurm/0003_umap_MIMICIV_records.slurm ${MODELS[$i]} 32 256 0.1
  sbatch slurm/0003_umap_MIMICIV_records.slurm ${MODELS[$i]} 32 256 0.01
  sbatch slurm/0003_umap_MIMICIV_records.slurm ${MODELS[$i]} 32 256 0.001
  sbatch slurm/0003_umap_MIMICIV_records.slurm ${MODELS[$i]} 64 256 0.1
  sbatch slurm/0003_umap_MIMICIV_records.slurm ${MODELS[$i]} 64 256 0.01
  sbatch slurm/0003_umap_MIMICIV_records.slurm ${MODELS[$i]} 64 256 0.001
  sbatch slurm/0003_umap_MIMICIV_records.slurm ${MODELS[$i]} 128 256 0.1
  sbatch slurm/0003_umap_MIMICIV_records.slurm ${MODELS[$i]} 128 256 0.01
  sbatch slurm/0003_umap_MIMICIV_records.slurm ${MODELS[$i]} 128 256 0.001
  sbatch slurm/0003_umap_MIMICIV_records.slurm ${MODELS[$i]} 256 256 0.1
  sbatch slurm/0003_umap_MIMICIV_records.slurm ${MODELS[$i]} 256 256 0.01
  sbatch slurm/0003_umap_MIMICIV_records.slurm ${MODELS[$i]} 256 256 0.001
done
