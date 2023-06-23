#-
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#-
SEED = 666142
CACHEDIR = "/global/cfs/cdirs/m1532/Projects_MVP/_models/LLMs/huggingface_cache/"
MAXLEN_BioBart_base = 1024 #- from: https://huggingface.co/GanjinZero/biobart-v2-base/blob/main/config.json
MAXLEN_BioBart_large = 1024 #- from: https://huggingface.co/GanjinZero/biobart-v2-large/blob/main/config.json
LOADLIMIT = 128
DEBUGSET = ["Patient had a heart attack on their way to work.", 
            "Patient had a heart attack on their way to work."]

#--
def read_BioBART_base():
  tokenizer = AutoTokenizer.from_pretrained("GanjinZero/biobart-v2-base", cache_dir=CACHEDIR)
  model = AutoModelForSeq2SeqLM.from_pretrained("GanjinZero/biobart-v2-base", cache_dir=CACHEDIR)
  return tokenizer, model

#--
def read_BioBART_large():
  tokenizer = AutoTokenizer.from_pretrained("GanjinZero/biobart-v2-large", cache_dir=CACHEDIR)
  model = AutoModelForSeq2SeqLM.from_pretrained("GanjinZero/biobart-v2-large", cache_dir=CACHEDIR)
  return tokenizer, model

#--
if __name__ == "__main__" :

  #- 
  from utils.parallel import SLURMDistributedTorch
  from utils.test import benchmark_llm

  #-
  with SLURMDistributedTorch(seed=SEED) as context:
    
    #- Benchmark LLM Runtimes
    benchmark_llm("BioBART_base", read_BioBART_base, LOADLIMIT, MAXLEN_BioBart_base, DEBUGSET, context)
    benchmark_llm("BioBART_large", read_BioBART_large, LOADLIMIT, MAXLEN_BioBart_large, DEBUGSET, context)
    