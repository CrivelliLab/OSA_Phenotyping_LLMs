#- 
from transformers import AutoTokenizer, T5ForConditionalGeneration

#-
SEED = 666142
CACHEDIR = "/global/cfs/cdirs/m1532/Projects_MVP/_models/LLMs/huggingface_cache/"
MAXLEN_ClinicalT5_base = 1024 #- from: https://huggingface.co/luqh/ClinicalT5-base/blob/main/config.json
MAXLEN_ClinicalT5_large = 1024 #- from: https://huggingface.co/luqh/ClinicalT5-large/blob/main/config.json
LOADLIMIT = 128
DEBUGSET = ["Patient had a heart attack on their way to work.", 
            "Patient had a heart attack on their way to work."]

#--
def read_ClinicalT5_base():
  tokenizer = AutoTokenizer.from_pretrained("luqh/ClinicalT5-base", cache_dir=CACHEDIR)
  model = T5ForConditionalGeneration.from_pretrained("luqh/ClinicalT5-base", from_flax=True, cache_dir=CACHEDIR)
  return tokenizer, model

#--
def read_ClinicalT5_large():
  tokenizer = AutoTokenizer.from_pretrained("luqh/ClinicalT5-large", cache_dir=CACHEDIR)
  model = T5ForConditionalGeneration.from_pretrained("luqh/ClinicalT5-large", from_flax=True, cache_dir=CACHEDIR)
  return tokenizer, model

#--
if __name__ == "__main__" :

  #-
  from utils.parallel import SLURMDistributedTorch
  from utils.test import benchmark_llm

  #-
  with SLURMDistributedTorch(seed=SEED) as context:

    #- Benchmark LLM Runtimes
    benchmark_llm("ClinicalT5_base", read_ClinicalT5_base, LOADLIMIT, MAXLEN_ClinicalT5_base, DEBUGSET, context)
    benchmark_llm("ClinicalT5_large", read_ClinicalT5_large, LOADLIMIT, MAXLEN_ClinicalT5_large, DEBUGSET, context)