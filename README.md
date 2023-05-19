# OSA Phenotyping LLMs
Developing analytic tools for phenotyping Obstructive Sleep Apnea along clinical text.

![OSA_WORDCLOUD](data/sleepapnea.png)

---

## Dataset

We are using MIMIC-IV to study the phenotyping of obstructive sleep apnea (OSA) in ICU visits.
On NERSC, a local copy was pulled and stored under `/project/projectdirs/m1532/Projects_MVP/_datasets/MIMIC_IV`. The file structure of the dataset is as follows:

``` 
# tree -h MIMIC_IV/
.
└── [4.0K]  physionet.org
    ├── [4.0K]  files
    │   ├── [4.0K]  mimiciv
    │   │   └── [4.0K]  2.2
    │   │       ├── [ 13K]  CHANGELOG.txt
    │   │       ├── [4.0K]  hosp
    │   │       │   ├── [ 70M]  admissions.csv
    │   │       │   ├── [3.2M]  d_hcpcs.csv
    │   │       │   ├── [129M]  diagnoses_icd.csv
    │   │       │   ├── [8.4M]  d_icd_diagnoses.csv
    │   │       │   ├── [7.0M]  d_icd_procedures.csv
    │   │       │   ├── [ 62K]  d_labitems.csv
    │   │       │   ├── [ 40M]  drgcodes.csv
    │   │       │   ├── [3.6G]  emar.csv
    │   │       │   ├── [5.1G]  emar_detail.csv
    │   │       │   ├── [9.2M]  hcpcsevents.csv
    │   │       │   ├── [2.8K]  index.html
    │   │       │   ├── [ 13G]  labevents.csv
    │   │       │   ├── [707M]  microbiologyevents.csv
    │   │       │   ├── [254M]  omr.csv
    │   │       │   ├── [9.4M]  patients.csv
    │   │       │   ├── [2.8G]  pharmacy.csv
    │   │       │   ├── [3.6G]  poe.csv
    │   │       │   ├── [214M]  poe_detail.csv
    │   │       │   ├── [2.5G]  prescriptions.csv
    │   │       │   ├── [ 25M]  procedures_icd.csv
    │   │       │   ├── [277K]  provider.csv
    │   │       │   ├── [ 20M]  services.csv
    │   │       │   └── [150M]  transfers.csv
    │   │       ├── [4.0K]  icu
    │   │       │   ├── [ 89K]  caregiver.csv
    │   │       │   ├── [ 28G]  chartevents.csv
    │   │       │   ├── [742M]  datetimeevents.csv
    │   │       │   ├── [360K]  d_items.csv
    │   │       │   ├── [ 11M]  icustays.csv
    │   │       │   ├── [1.3K]  index.html
    │   │       │   ├── [1.9G]  ingredientevents.csv
    │   │       │   ├── [2.2G]  inputevents.csv
    │   │       │   ├── [349M]  outputevents.csv
    │   │       │   └── [123M]  procedureevents.csv
    │   │       ├── [ 789]  index.html
    │   │       ├── [2.5K]  LICENSE.txt
    │   │       └── [2.8K]  SHA256SUMS.txt
    │   └── [4.0K]  mimic-iv-note
    │       └── [4.0K]  2.2
    │           ├── [ 574]  index.html
    │           ├── [2.5K]  LICENSE.txt
    │           ├── [4.0K]  note
    │           │   ├── [1.1G]  discharge.csv.gz
    │           │   ├── [1.3M]  discharge_detail.csv.gz
    │           │   ├── [ 737]  index.html
    │           │   ├── [746M]  radiology.csv.gz
    │           │   └── [ 37M]  radiology_detail.csv.gz
    │           └── [ 439]  SHA256SUMS.txt
    └── [  22]  robots.txt

9 directories, 46 files
```

## Language Models

For this analysis, we explore the representation quality of several contempory language models.
10 models have been stored locally on NERSC under `/project/projectdirs/m1532/Projects_MVP/_models/LLMs`.
The file structure for the LLMs is as follows:

```
tree -h LLMs/

.
├── [4.0K]  biobart-base
│   └── [4.0K]  models--GanjinZero--biobart-base
│       ├── [4.0K]  blobs
│       │   ├── [878K]  0a39732b2d8be8e493cab3da68b68cc3e28221de
│       │   ├── [446K]  226b0752cac7789c48f0cb3ec53eda48b7be36cc
│       │   ├── [1.7K]  244c5264610c5aa286155e7e0049e21801ecae2b
│       │   ├── [ 576]  244d03ff410dd77b93e4fc282d3bdc8dab79d1cf
│       │   ├── [1.2K]  2bf99854609538f90af03c921c41a0cbde670b2f
│       │   ├── [266M]  53826e5fa79faa6b9ea4f0b36f44afa6ec3c98314be1bbe35b94ae8c3af057f3
│       │   ├── [266M]  8e6d2e959b434fc3f3c0814fa2d46d74621f43d6b6da6b1d30acf39aa8cbf945
│       │   ├── [ 772]  e97d1993365bb21c88f390e8703e4c1af564821f
│       │   └── [1.0K]  fb52f2335cf19966cd95e48af666d27f1370e532
│       ├── [4.0K]  refs
│       │   └── [  40]  main
│       └── [4.0K]  snapshots
│           └── [4.0K]  5e85794b3826842a73d7abacf5e2b1b64dbfdcb1
│               ├── [  52]  config.json -> ../../blobs/244c5264610c5aa286155e7e0049e21801ecae2b
│               ├── [  52]  merges.txt -> ../../blobs/226b0752cac7789c48f0cb3ec53eda48b7be36cc
│               ├── [  76]  model.safetensors -> ../../blobs/8e6d2e959b434fc3f3c0814fa2d46d74621f43d6b6da6b1d30acf39aa8cbf945
│               ├── [  76]  pytorch_model.bin -> ../../blobs/53826e5fa79faa6b9ea4f0b36f44afa6ec3c98314be1bbe35b94ae8c3af057f3
│               ├── [  52]  README.md -> ../../blobs/244d03ff410dd77b93e4fc282d3bdc8dab79d1cf
│               ├── [  52]  special_tokens_map.json -> ../../blobs/e97d1993365bb21c88f390e8703e4c1af564821f
│               ├── [  52]  tokenizer_config.json -> ../../blobs/fb52f2335cf19966cd95e48af666d27f1370e532
│               └── [  52]  vocab.json -> ../../blobs/0a39732b2d8be8e493cab3da68b68cc3e28221de
├── [4.0K]  biobart-large
│   └── [4.0K]  models--GanjinZero--biobart-large
│       ├── [4.0K]  blobs
│       │   ├── [878K]  0a39732b2d8be8e493cab3da68b68cc3e28221de
│       │   ├── [446K]  226b0752cac7789c48f0cb3ec53eda48b7be36cc
│       │   ├── [ 576]  244d03ff410dd77b93e4fc282d3bdc8dab79d1cf
│       │   ├── [1.2K]  2bf99854609538f90af03c921c41a0cbde670b2f
│       │   ├── [1.7K]  5d4bde8a2817be97761ea549b88301d65194aeb3
│       │   ├── [1.1K]  76cedc0dfba570e327daf72ac283bae67c33fbe4
│       │   ├── [775M]  a073e8ecba4aaa5c7830b0db1e1816ffa14b637b4098caae8b861166c34ed26c
│       │   ├── [775M]  c8e153793fe9728d25ab2742ac118cc6fbd537df3dace510f605255632ce8bbc
│       │   └── [ 772]  e97d1993365bb21c88f390e8703e4c1af564821f
│       ├── [4.0K]  refs
│       │   └── [  40]  main
│       └── [4.0K]  snapshots
│           └── [4.0K]  47c968d3aba260d5737fda6f5b2ad3aa35ce75e8
│               ├── [  52]  config.json -> ../../blobs/5d4bde8a2817be97761ea549b88301d65194aeb3
│               ├── [  52]  merges.txt -> ../../blobs/226b0752cac7789c48f0cb3ec53eda48b7be36cc
│               ├── [  76]  model.safetensors -> ../../blobs/a073e8ecba4aaa5c7830b0db1e1816ffa14b637b4098caae8b861166c34ed26c
│               ├── [  76]  pytorch_model.bin -> ../../blobs/c8e153793fe9728d25ab2742ac118cc6fbd537df3dace510f605255632ce8bbc
│               ├── [  52]  README.md -> ../../blobs/244d03ff410dd77b93e4fc282d3bdc8dab79d1cf
│               ├── [  52]  special_tokens_map.json -> ../../blobs/e97d1993365bb21c88f390e8703e4c1af564821f
│               ├── [  52]  tokenizer_config.json -> ../../blobs/76cedc0dfba570e327daf72ac283bae67c33fbe4
│               └── [  52]  vocab.json -> ../../blobs/0a39732b2d8be8e493cab3da68b68cc3e28221de
├── [4.0K]  biobert_large
│   ├── [ 289]  bert_config_bio_58k_large.json
│   ├── [1.4G]  bio_bert_large_1000k.ckpt.data-00000-of-00001
│   ├── [ 16K]  bio_bert_large_1000k.ckpt.index
│   ├── [1.6M]  bio_bert_large_1000k.ckpt.meta
│   └── [456K]  vocab_cased_pubmed_pmc_30k.txt
├── [4.0K]  biobert_v1.1_pubmed
│   ├── [ 313]  bert_config.json
│   ├── [413M]  model.ckpt-1000000.data-00000-of-00001
│   ├── [8.0K]  model.ckpt-1000000.index
│   ├── [925K]  model.ckpt-1000000.meta
│   └── [208K]  vocab.txt
├── [4.0K]  downloads
│   ├── [816M]  biobert_large_v1.1_pubmed.tar.gz
│   ├── [383M]  biobert_v1.1_pubmed.tar.gz
│   ├── [2.6G]  Clinical-T5-Large.bin.gz
│   ├── [789M]  Clinical-T5-Sci.bin.gz
│   ├── [789M]  Clinical-T5-Scratch.bin.gz
│   ├── [5.0G]  gatortron_og_1.zip
│   ├── [5.0G]  gatortron_s_1.zip
│   ├── [ 14G]  Pre-trained-BioGPT-Large.tgz
│   └── [3.2G]  Pre-trained-BioGPT.tgz
├── [4.0K]  gatortron_og_1
│   ├── [ 150]  config.json
│   ├── [ 918]  hparam.yaml
│   ├── [788M]  MegatronBERT.nemo
│   ├── [4.6G]  MegatronBERT.pt
│   └── [370K]  vocab.txt
├── [4.0K]  gatortron_s_1
│   ├── [ 150]  config.json
│   ├── [ 918]  hparam.yaml
│   ├── [787M]  MegatronBERT.nemo
│   ├── [4.6G]  MegatronBERT.pt
│   └── [370K]  vocab.txt
├── [4.0K]  megatron-bert-uncased-345m
│   └── [592M]  checkpoint.zip
├── [4.0K]  Pre-trained-BioGPT
│   └── [4.0G]  checkpoint.pt
└── [4.0K]  Pre-trained-BioGPT-Large
    └── [ 18G]  checkpoint.pt

```

### Papers:
The following are the papers for these models.

* BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining [[Paper]](https://arxiv.org/pdf/2210.10341.pdf) [[GitHub]](https://github.com/microsoft/BioGPT)

* BioBERT: a pre-trained biomedical language representation model for biomedical text mining [[Paper]](https://arxiv.org/pdf/1901.08746.pdf) [[GitHub]](https://github.com/dmis-lab/biobert)

* BioBART: Pretraining and Evaluation of A Biomedical Generative Language Model [[Paper]](https://aclanthology.org/2022.bionlp-1.9.pdf) [[GitHub]](https://github.com/GanjinZero/BioBART)

* BioMegatron: Larger Biomedical Domain Language Model [[Paper]](https://arxiv.org/pdf/2010.06060.pdf) [[HuggingFace]](https://huggingface.co/nvidia/megatron-bert-uncased-345m)

* ClinicalT5: A Generative Language Model for Clinical Text [[Paper]](https://aclanthology.org/2022.findings-emnlp.398.pdf) [[Repo]](https://physionet.org/content/clinical-t5/1.0.0/)

* GatorTron: A Large Clinical Language Model to Unlock Patient Information from Unstructured Electronic Health Records [[Paper]](https://arxiv.org/pdf/2203.03540v2.pdf) [[Paper]](https://www.nature.com/articles/s41746-022-00742-2) [[Catalog]](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/gatortron_s) [[Catalog]](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/gatortron_og)

* RadBERT: Adapting Transformer-based Language Models to Radiology [[paper]](https://pubs.rsna.org/doi/epdf/10.1148/ryai.210258) [[HuggingFace]](https://huggingface.co/UCSD-VA-health/RadBERT-RoBERTa-4m) [[GitHub]](https://github.com/zzxslp/RadBERT)

## Analysis Workflow

> Previous work from summer 2022 is in repo under `src/old`. It contains scripts used for
building/searching MIMIC-III vocabs and patient cohorts by Dxs. Also, it contains the script to
fit a logistic regression model against presence/absence NLP variables.

For this analysis, we want to explore the how different pretrained large language models (LLMs) represent patient documents and observe if their is strong clustering w.r.t. commorbidities and outcome progression.
With this goal in mind, we will build tools to process MIMIC-IV notes with language models,
visualize the latent space of documents and map the resulting embeddings to longitudinal data.
The overall structure of the analysis is as follows:

1. Subset and label patient data by OSA-related dx.
2. Embed documents using LLMs.
3. Use UMAP to visualize latent pace and plot structured variables.
4. Measure correlation between clustering diversity and structured variables.

The following are the scripts used in the main workflow of the analysis:

```
.
├── LICENSE
├── README.md
├── data
└── src
    ├── 0_subset_MIMICIV_phenotype.py
    ├── 1_embedd_MIMICIV_records.py
    ├── 2_umap2D_MIMICIV_latent.py
    ├── 3_clustr_MIMICIV_latent.py
```

## Notes

- `discharge.csv.gz` contains 331,794 reports for 145,915 patients. The mean number of charcters per document is 10,551. The longest and shortest documents had 60,381 and 353 charcters respectively. 

- After lowercasing text, 13,519 matches on r"\sosa\s" and 6,672 matches on "\sobstructive sleep apnea\s".
