#-
MODELS=( \
  #-- Layer 0
  # "output/embeds/BioBART_base.discharge.layer.0/" \
  # "output/embeds/BioBART_large.discharge.layer.0/" \
  # "output/embeds/BioBERT_base.discharge.layer.0/" \
  # "output/embeds/BioBERT_large.discharge.layer.0/" \
  # "output/embeds/BioGPT_base.discharge.layer.0/" \
  # "output/embeds/BioGPT_large.discharge.layer.0/" \
  # "output/embeds/BioMegatron_base.discharge.layer.0/" \
  # "output/embeds/Bio_ClinicalBERT.discharge.layer.0/" \
  # "output/embeds/Gatortron_base.discharge.layer.0/" \
  # "output/embeds/Gatortron_s.discharge.layer.0/" \
  # "output/embeds/Gatortron_medium.discharge.layer.0/" \
  # "output/embeds/RadBERT_2m.discharge.layer.0/" \
  # "output/embeds/RadBERT_4m.discharge.layer.0/" \

  #-- Layer 1
  "output/embeds/BioBART_base.discharge.layer.1/" \
  "output/embeds/BioBART_large.discharge.layer.1/" \
  "output/embeds/BioBERT_base.discharge.layer.1/" \
  "output/embeds/BioBERT_large.discharge.layer.1/" \
  "output/embeds/BioGPT_base.discharge.layer.1/" \
  "output/embeds/BioGPT_large.discharge.layer.1/" \
  "output/embeds/BioMegatron_base.discharge.layer.1/" \
  "output/embeds/Bio_ClinicalBERT.discharge.layer.1/" \
  "output/embeds/Gatortron_base.discharge.layer.1/" \
  "output/embeds/Gatortron_s.discharge.layer.1/" \
  "output/embeds/Gatortron_medium.discharge.layer.1/" \
  "output/embeds/RadBERT_2m.discharge.layer.1/" \
  # "output/embeds/RadBERT_4m.discharge.layer.1/" \
)

#- KMeans Clustering Jobs
for i in {0..11} 
do 
  sbatch slurm/0002_clustr_MIMICIV_records.slurm ${MODELS[$i]} 32 
  sbatch slurm/0002_clustr_MIMICIV_records.slurm ${MODELS[$i]} 64 
  sbatch slurm/0002_clustr_MIMICIV_records.slurm ${MODELS[$i]} 128 
  sbatch slurm/0002_clustr_MIMICIV_records.slurm ${MODELS[$i]} 256
  sbatch slurm/0002_clustr_MIMICIV_records.slurm ${MODELS[$i]} 512
  sbatch slurm/0002_clustr_MIMICIV_records.slurm ${MODELS[$i]} 1024

done

