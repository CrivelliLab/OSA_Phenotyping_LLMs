#-
from transformers import AutoTokenizer, AutoModel

#-
SEED = 666142
CACHEDIR = "/global/cfs/cdirs/m1532/Projects_MVP/_models/LLMs/huggingface_cache/"
MAXLEN_Bio_ClinicalBERT = 512 #- from: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/blob/main/config.json
LOADLIMIT = 128
DEBUGSET = ["Patient had a heart attack on their way to work.", 
            "Patient had a heart attack on their way to work."]

#--
def read_Bio_ClinicalBERT():
  tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", cache_dir=CACHEDIR)
  model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", cache_dir=CACHEDIR)
  return tokenizer, model

#--
if __name__ == "__main__" :

  #-
  from utils.parallel import SLURMDistributedTorch
  from utils.test import benchmark_llm

  #-
  with SLURMDistributedTorch(seed=SEED) as context:

    #- Benchmark LLM Runtimes
    benchmark_llm("Bio_ClinicalBERT", read_Bio_ClinicalBERT, LOADLIMIT, MAXLEN_Bio_ClinicalBERT, DEBUGSET, context)
