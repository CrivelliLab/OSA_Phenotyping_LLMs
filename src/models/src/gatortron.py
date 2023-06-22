#- 
from transformers import AutoTokenizer, AutoModel
from utils.parallel import SLURMDistributedTorch
from utils.test import benchmark_llm

#-
SEED = 666142
CACHEDIR = "/global/cfs/cdirs/m1532/Projects_MVP/_models/LLMs/huggingface_cache/"
MAXLEN_Gatortron_base = 512 #- from: https://huggingface.co/UFNLP/gatortron-base/blob/main/config.json
MAXLEN_Gatortron_s = 512 #- from: https://huggingface.co/UFNLP/gatortronS/blob/main/config.json
MAXLEN_Gatortron_medium = 512 #- from: https://huggingface.co/UFNLP/gatortron-medium/blob/main/config.json
LOADLIMIT = 128
DEBUGSET = ["Patient had a heart attack on their way to work.", 
            "Patient had a heart attack on their way to work."]

#--
def read_Gatortron_base():
  tokenizer = AutoTokenizer.from_pretrained("UFNLP/gatortron-base", cache_dir=CACHEDIR)
  model = AutoModel.from_pretrained("UFNLP/gatortron-base", cache_dir=CACHEDIR)
  return tokenizer, model

#--
def read_Gatortron_s():
  tokenizer = AutoTokenizer.from_pretrained("UFNLP/gatortronS", cache_dir=CACHEDIR)
  model = AutoModel.from_pretrained("UFNLP/gatortronS", cache_dir=CACHEDIR)
  return tokenizer, model

#--
def read_Gatortron_medium():
  tokenizer = AutoTokenizer.from_pretrained("UFNLP/gatortron-medium", cache_dir=CACHEDIR)
  model = AutoModel.from_pretrained("UFNLP/gatortron-medium", cache_dir=CACHEDIR)
  return tokenizer, model

#--
if __name__ == "__main__" :

  #-
  with SLURMDistributedTorch(seed=SEED) as context:

    #- Benchmark LLM Runtimes
    benchmark_llm("Gatortron_base", read_Gatortron_base, LOADLIMIT, MAXLEN_Gatortron_base, DEBUGSET, context)
    benchmark_llm("Gatortron_s", read_Gatortron_s, LOADLIMIT, MAXLEN_Gatortron_s, DEBUGSET, context)
    benchmark_llm("Gatortron_medium", read_Gatortron_medium, LOADLIMIT, MAXLEN_Gatortron_medium, DEBUGSET, context)
    