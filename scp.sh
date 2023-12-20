#-- ssh login
#scp rzamora@perlmutter-p1.nersc.gov:/pscratch/sd/r/rzamora/OSA_Phenotyping_LLMs/output/embeds/BioBART_base.discharge.layer.0/umap/*  output/
#scp rzamora@perlmutter-p1.nersc.gov:/pscratch/sd/r/rzamora/OSA_Phenotyping_LLMs/notebooks/*  notebooks/
# scp rzamora@perlmutter-p1.nersc.gov:/pscratch/sd/r/rzamora/OSA_Phenotyping_LLMs/src/*.py src/
# scp rzamora@perlmutter-p1.nersc.gov:/pscratch/sd/r/rzamora/OSA_Phenotyping_LLMs/slurm/* slurm/
# scp rzamora@perlmutter-p1.nersc.gov:/pscratch/sd/r/rzamora/OSA_Phenotyping_LLMs/*.sh .
scp data/dxs/* rzamora@perlmutter-p1.nersc.gov:/pscratch/sd/r/rzamora/OSA_Phenotyping_LLMs/data/dxs/

├── [4.0K]  BioBART_base.discharge.layer.0
│   ├── [4.0K]  kmeans
│   │   ├── [1.3G]  nclusters.1024.pt
│   │   ├── [162M]  nclusters.128.pt
│   │   ├── [324M]  nclusters.256.pt
│   │   ├── [ 41M]  nclusters.32.pt
│   │   ├── [648M]  nclusters.512.pt
│   │   └── [ 81M]  nclusters.64.pt
│   └── [4.0K]  umap
│       ├── [ 29M]  nclusters.1024.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.512.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.64.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.64.nn.256.min.0.01.csv
│       └── [ 29M]  nclusters.64.nn.256.min.0.1.csv
├── [4.0K]  BioBART_base.discharge.layer.1
│   ├── [4.0K]  kmeans
│   │   ├── [1.3G]  nclusters.1024.pt
│   │   ├── [162M]  nclusters.128.pt
│   │   ├── [324M]  nclusters.256.pt
│   │   ├── [ 41M]  nclusters.32.pt
│   │   ├── [648M]  nclusters.512.pt
│   │   └── [ 81M]  nclusters.64.pt
│   └── [4.0K]  umap
│       ├── [ 29M]  nclusters.1024.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.512.nn.256.min.0.001.csv
│       └── [ 29M]  nclusters.64.nn.256.min.0.001.csv
├── [4.0K]  BioBART_large.discharge.layer.0
│   ├── [4.0K]  kmeans
│   │   ├── [1.3G]  nclusters.1024.pt
│   │   ├── [162M]  nclusters.128.pt
│   │   ├── [324M]  nclusters.256.pt
│   │   ├── [ 41M]  nclusters.32.pt
│   │   ├── [648M]  nclusters.512.pt
│   │   └── [ 81M]  nclusters.64.pt
│   └── [4.0K]  umap
│       ├── [ 29M]  nclusters.1024.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.512.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.64.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.64.nn.256.min.0.01.csv
│       └── [ 29M]  nclusters.64.nn.256.min.0.1.csv
├── [4.0K]  BioBART_large.discharge.layer.1
│   ├── [4.0K]  kmeans
│   │   ├── [1.3G]  nclusters.1024.pt
│   │   ├── [162M]  nclusters.128.pt
│   │   ├── [324M]  nclusters.256.pt
│   │   ├── [ 41M]  nclusters.32.pt
│   │   ├── [648M]  nclusters.512.pt
│   │   └── [ 81M]  nclusters.64.pt
│   └── [4.0K]  umap
│       ├── [ 29M]  nclusters.1024.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.512.nn.256.min.0.001.csv
│       └── [ 29M]  nclusters.64.nn.256.min.0.001.csv
├── [4.0K]  BioBERT_base.discharge.layer.0
│   ├── [4.0K]  kmeans
│   │   ├── [1.3G]  nclusters.1024.pt
│   │   ├── [162M]  nclusters.128.pt
│   │   ├── [324M]  nclusters.256.pt
│   │   ├── [ 41M]  nclusters.32.pt
│   │   ├── [648M]  nclusters.512.pt
│   │   └── [ 81M]  nclusters.64.pt
│   └── [4.0K]  umap
│       ├── [ 29M]  nclusters.1024.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.512.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.64.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.64.nn.256.min.0.01.csv
│       └── [ 29M]  nclusters.64.nn.256.min.0.1.csv
├── [4.0K]  BioBERT_base.discharge.layer.1
│   ├── [4.0K]  kmeans
│   │   ├── [1.3G]  nclusters.1024.pt
│   │   ├── [162M]  nclusters.128.pt
│   │   ├── [324M]  nclusters.256.pt
│   │   ├── [ 41M]  nclusters.32.pt
│   │   ├── [648M]  nclusters.512.pt
│   │   └── [ 81M]  nclusters.64.pt
│   └── [4.0K]  umap
│       ├── [ 29M]  nclusters.1024.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.512.nn.256.min.0.001.csv
│       └── [ 29M]  nclusters.64.nn.256.min.0.001.csv
├── [4.0K]  BioBERT_large.discharge.layer.0
│   ├── [4.0K]  kmeans
│   │   ├── [1.3G]  nclusters.1024.pt
│   │   ├── [162M]  nclusters.128.pt
│   │   ├── [324M]  nclusters.256.pt
│   │   ├── [ 41M]  nclusters.32.pt
│   │   ├── [648M]  nclusters.512.pt
│   │   └── [ 81M]  nclusters.64.pt
│   └── [4.0K]  umap
│       ├── [ 29M]  nclusters.1024.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.512.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.64.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.64.nn.256.min.0.01.csv
│       └── [ 29M]  nclusters.64.nn.256.min.0.1.csv
├── [4.0K]  BioBERT_large.discharge.layer.1
│   ├── [4.0K]  kmeans
│   │   ├── [1.3G]  nclusters.1024.pt
│   │   ├── [162M]  nclusters.128.pt
│   │   ├── [324M]  nclusters.256.pt
│   │   ├── [ 41M]  nclusters.32.pt
│   │   ├── [648M]  nclusters.512.pt
│   │   └── [ 81M]  nclusters.64.pt
│   └── [4.0K]  umap
│       ├── [ 29M]  nclusters.1024.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.512.nn.256.min.0.001.csv
│       └── [ 29M]  nclusters.64.nn.256.min.0.001.csv
├── [4.0K]  Bio_ClinicalBERT.discharge.layer.0
│   ├── [4.0K]  kmeans
│   │   ├── [1.3G]  nclusters.1024.pt
│   │   ├── [162M]  nclusters.128.pt
│   │   ├── [324M]  nclusters.256.pt
│   │   ├── [ 41M]  nclusters.32.pt
│   │   ├── [648M]  nclusters.512.pt
│   │   └── [ 81M]  nclusters.64.pt
│   └── [4.0K]  umap
│       ├── [ 29M]  nclusters.1024.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.512.nn.256.min.0.001.csv
│       └── [ 29M]  nclusters.64.nn.256.min.0.001.csv
├── [4.0K]  Bio_ClinicalBERT.discharge.layer.1
│   └── [4.0K]  kmeans
│       ├── [1.3G]  nclusters.1024.pt
│       ├── [162M]  nclusters.128.pt
│       ├── [324M]  nclusters.256.pt
│       ├── [ 41M]  nclusters.32.pt
│       ├── [648M]  nclusters.512.pt
│       └── [ 81M]  nclusters.64.pt
├── [4.0K]  BioGPT_base.discharge.layer.0
│   ├── [4.0K]  kmeans
│   │   ├── [1.3G]  nclusters.1024.pt
│   │   ├── [162M]  nclusters.128.pt
│   │   ├── [324M]  nclusters.256.pt
│   │   ├── [ 41M]  nclusters.32.pt
│   │   ├── [648M]  nclusters.512.pt
│   │   └── [ 81M]  nclusters.64.pt
│   └── [4.0K]  umap
│       ├── [ 29M]  nclusters.1024.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.512.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.64.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.64.nn.256.min.0.01.csv
│       └── [ 29M]  nclusters.64.nn.256.min.0.1.csv
├── [4.0K]  BioGPT_base.discharge.layer.1
│   ├── [4.0K]  kmeans
│   │   ├── [1.3G]  nclusters.1024.pt
│   │   ├── [162M]  nclusters.128.pt
│   │   ├── [324M]  nclusters.256.pt
│   │   ├── [ 41M]  nclusters.32.pt
│   │   ├── [648M]  nclusters.512.pt
│   │   └── [ 81M]  nclusters.64.pt
│   └── [4.0K]  umap
│       ├── [ 29M]  nclusters.1024.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.512.nn.256.min.0.001.csv
│       └── [ 29M]  nclusters.64.nn.256.min.0.001.csv
├── [4.0K]  BioGPT_large.discharge.layer.0
│   ├── [4.0K]  kmeans
│   │   ├── [1.3G]  nclusters.1024.pt
│   │   ├── [162M]  nclusters.128.pt
│   │   ├── [324M]  nclusters.256.pt
│   │   ├── [ 41M]  nclusters.32.pt
│   │   ├── [648M]  nclusters.512.pt
│   │   └── [ 81M]  nclusters.64.pt
│   └── [4.0K]  umap
│       ├── [ 29M]  nclusters.1024.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.512.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.64.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.64.nn.256.min.0.01.csv
│       └── [ 29M]  nclusters.64.nn.256.min.0.1.csv
├── [4.0K]  BioGPT_large.discharge.layer.1
│   ├── [4.0K]  kmeans
│   │   ├── [1.3G]  nclusters.1024.pt
│   │   ├── [162M]  nclusters.128.pt
│   │   ├── [324M]  nclusters.256.pt
│   │   ├── [ 41M]  nclusters.32.pt
│   │   ├── [648M]  nclusters.512.pt
│   │   └── [ 81M]  nclusters.64.pt
│   └── [4.0K]  umap
│       ├── [ 29M]  nclusters.128.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.001.csv
│       └── [ 29M]  nclusters.64.nn.256.min.0.001.csv
├── [4.0K]  BioMegatron_base.discharge.layer.0
│   ├── [4.0K]  kmeans
│   │   ├── [1.3G]  nclusters.1024.pt
│   │   ├── [162M]  nclusters.128.pt
│   │   ├── [324M]  nclusters.256.pt
│   │   ├── [ 41M]  nclusters.32.pt
│   │   ├── [648M]  nclusters.512.pt
│   │   └── [ 81M]  nclusters.64.pt
│   └── [4.0K]  umap
│       ├── [ 29M]  nclusters.1024.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.512.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.64.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.64.nn.256.min.0.01.csv
│       └── [ 29M]  nclusters.64.nn.256.min.0.1.csv
├── [4.0K]  BioMegatron_base.discharge.layer.1
│   └── [4.0K]  kmeans
│       ├── [1.3G]  nclusters.1024.pt
│       ├── [162M]  nclusters.128.pt
│       ├── [324M]  nclusters.256.pt
│       ├── [ 41M]  nclusters.32.pt
│       ├── [648M]  nclusters.512.pt
│       └── [ 81M]  nclusters.64.pt
├── [4.0K]  Gatortron_base.discharge.layer.0
│   ├── [4.0K]  kmeans
│   │   ├── [1.3G]  nclusters.1024.pt
│   │   ├── [162M]  nclusters.128.pt
│   │   ├── [324M]  nclusters.256.pt
│   │   ├── [ 41M]  nclusters.32.pt
│   │   ├── [648M]  nclusters.512.pt
│   │   └── [ 81M]  nclusters.64.pt
│   └── [4.0K]  umap
│       ├── [ 29M]  nclusters.1024.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.512.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.64.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.64.nn.256.min.0.01.csv
│       └── [ 29M]  nclusters.64.nn.256.min.0.1.csv
├── [4.0K]  Gatortron_base.discharge.layer.1
│   └── [4.0K]  kmeans
│       ├── [1.3G]  nclusters.1024.pt
│       ├── [162M]  nclusters.128.pt
│       ├── [324M]  nclusters.256.pt
│       ├── [ 41M]  nclusters.32.pt
│       ├── [648M]  nclusters.512.pt
│       └── [ 81M]  nclusters.64.pt
├── [4.0K]  Gatortron_medium.discharge.layer.0
│   ├── [4.0K]  kmeans
│   │   ├── [1.3G]  nclusters.1024.pt
│   │   ├── [162M]  nclusters.128.pt
│   │   ├── [324M]  nclusters.256.pt
│   │   ├── [ 41M]  nclusters.32.pt
│   │   ├── [648M]  nclusters.512.pt
│   │   └── [ 81M]  nclusters.64.pt
│   └── [4.0K]  umap
│       ├── [ 29M]  nclusters.1024.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.512.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.64.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.64.nn.256.min.0.01.csv
│       └── [ 29M]  nclusters.64.nn.256.min.0.1.csv
├── [4.0K]  Gatortron_medium.discharge.layer.1
│   └── [4.0K]  kmeans
│       ├── [1.3G]  nclusters.1024.pt
│       ├── [162M]  nclusters.128.pt
│       ├── [324M]  nclusters.256.pt
│       ├── [ 41M]  nclusters.32.pt
│       ├── [648M]  nclusters.512.pt
│       └── [ 81M]  nclusters.64.pt
├── [4.0K]  Gatortron_s.discharge.layer.0
│   ├── [4.0K]  kmeans
│   │   ├── [1.3G]  nclusters.1024.pt
│   │   ├── [162M]  nclusters.128.pt
│   │   ├── [324M]  nclusters.256.pt
│   │   ├── [ 41M]  nclusters.32.pt
│   │   ├── [648M]  nclusters.512.pt
│   │   └── [ 81M]  nclusters.64.pt
│   └── [4.0K]  umap
│       ├── [ 29M]  nclusters.1024.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.512.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.64.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.64.nn.256.min.0.01.csv
│       └── [ 29M]  nclusters.64.nn.256.min.0.1.csv
├── [4.0K]  Gatortron_s.discharge.layer.1
│   └── [4.0K]  kmeans
│       ├── [1.3G]  nclusters.1024.pt
│       ├── [162M]  nclusters.128.pt
│       ├── [324M]  nclusters.256.pt
│       ├── [ 41M]  nclusters.32.pt
│       ├── [648M]  nclusters.512.pt
│       └── [ 81M]  nclusters.64.pt
├── [4.0K]  RadBERT_2m.discharge.layer.0
│   ├── [4.0K]  kmeans
│   │   ├── [1.3G]  nclusters.1024.pt
│   │   ├── [162M]  nclusters.128.pt
│   │   ├── [324M]  nclusters.256.pt
│   │   ├── [ 41M]  nclusters.32.pt
│   │   ├── [648M]  nclusters.512.pt
│   │   └── [ 81M]  nclusters.64.pt
│   └── [4.0K]  umap
│       ├── [ 29M]  nclusters.1024.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.128.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.256.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.01.csv
│       ├── [ 29M]  nclusters.32.nn.256.min.0.1.csv
│       ├── [ 29M]  nclusters.512.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.64.nn.256.min.0.001.csv
│       ├── [ 29M]  nclusters.64.nn.256.min.0.01.csv
│       └── [ 29M]  nclusters.64.nn.256.min.0.1.csv
├── [4.0K]  RadBERT_2m.discharge.layer.1
│   └── [4.0K]  kmeans
│       ├── [1.3G]  nclusters.1024.pt
│       ├── [162M]  nclusters.128.pt
│       ├── [324M]  nclusters.256.pt
│       ├── [ 41M]  nclusters.32.pt
│       ├── [648M]  nclusters.512.pt
│       └── [ 81M]  nclusters.64.pt
└── [4.0K]  RadBERT_4m.discharge.layer.0

67 directories, 338 files