#-
from transformers import AutoTokenizer, AutoModelForMaskedLM
from utils.parallel import SLURMDistributedTorch
from utils.test import benchmark_llm

#-
SEED = 666142
CACHEDIR = "/global/cfs/cdirs/m1532/Projects_MVP/_models/LLMs/huggingface_cache/"
MAXLEN_BioBert_base = 512 #- from: https://huggingface.co/dmis-lab/biobert-base-cased-v1.2/blob/main/config.json
MAXLEN_BioBert_large= 512 #- from: https://huggingface.co/dmis-lab/biobert-large-cased-v1.1/blob/main/config.json
LOADLIMIT = 128
DEBUGSET = ["Patient had a heart attack on their way to work.", 
            "Patient had a heart attack on their way to work."]

#--
def read_BioBERT_base():
  tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2", cache_dir=CACHEDIR)
  model = AutoModelForMaskedLM.from_pretrained("dmis-lab/biobert-base-cased-v1.2", cache_dir=CACHEDIR)
  return tokenizer, model

#--
def read_BioBERT_large():
  tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-large-cased-v1.1", cache_dir=CACHEDIR)
  model = AutoModelForMaskedLM.from_pretrained("dmis-lab/biobert-large-cased-v1.1", cache_dir=CACHEDIR)
  return tokenizer, model

#--
if __name__ == "__main__" :

  #-
  with SLURMDistributedTorch(seed=SEED) as context:

    #- Benchmark LLM Runtimes
    benchmark_llm("BioBERT_base", read_BioBERT_base, LOADLIMIT, MAXLEN_BioBert_base, DEBUGSET, context)
    benchmark_llm("BioBERT_large", read_BioBERT_large, LOADLIMIT, MAXLEN_BioBert_large, DEBUGSET, context)
