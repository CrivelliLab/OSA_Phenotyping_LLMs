#- 
from transformers import AutoTokenizer, AutoModelForMaskedLM
from utils.parallel import SLURMDistributedTorch
from utils.test import benchmark_llm

#-
SEED = 666142
CACHEDIR = "/global/cfs/cdirs/m1532/Projects_MVP/_models/LLMs/huggingface_cache/"
MAXLEN_RadBERT_2m = 512 #- https://huggingface.co/UCSD-VA-health/RadBERT-2m/blob/main/config.json
MAXLEN_RadBERT_4m = 514 #- https://huggingface.co/UCSD-VA-health/RadBERT-RoBERTa-4m/blob/main/config.json
LOADLIMIT = 128
DEBUGSET = ["Patient had a heart attack on their way to work.", 
            "Patient had a heart attack on their way to work."]

#--
def read_RadBERT_2m():
  tokenizer = AutoTokenizer.from_pretrained("UCSD-VA-health/RadBERT-2m", cache_dir=CACHEDIR)
  model = AutoModelForMaskedLM.from_pretrained("UCSD-VA-health/RadBERT-2m", cache_dir=CACHEDIR)
  return tokenizer, model

#--
def read_RadBERT_4m():
  tokenizer = AutoTokenizer.from_pretrained("UCSD-VA-health/RadBERT-RoBERTa-4m", cache_dir=CACHEDIR)
  model = AutoModelForMaskedLM.from_pretrained("UCSD-VA-health/RadBERT-RoBERTa-4m", cache_dir=CACHEDIR)
  return tokenizer, model

#--
if __name__ == "__main__" :

  #-
  with SLURMDistributedTorch(seed=SEED) as context:

    #- Benchmark LLM runtimes
    benchmark_llm("RadBERT_2m", read_RadBERT_2m, LOADLIMIT, MAXLEN_RadBERT_2m, DEBUGSET, context)
    benchmark_llm("RadBERT_4m", read_RadBERT_4m, LOADLIMIT, MAXLEN_RadBERT_4m, DEBUGSET, context)
   