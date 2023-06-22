#- 
from transformers import BioGptTokenizer, BioGptForCausalLM
from utils.parallel import SLURMDistributedTorch
from utils.test import benchmark_llm

#-
SEED = 666142
CACHEDIR = "/global/cfs/cdirs/m1532/Projects_MVP/_models/LLMs/huggingface_cache/"
MAXLEN_BioGPT_base = 1024 #- from: https://huggingface.co/docs/transformers/model_doc/biogpt#transformers.BioGptConfig
MAXLEN_BioGPT_large = 2048 #- from: https://huggingface.co/microsoft/BioGPT-Large/blob/main/config.json
LOADLIMIT = 128
DEBUGSET = ["Patient had a heart attack on their way to work.", 
            "Patient had a heart attack on their way to work."]

#--
def read_BioGPT_base():
  model = BioGptForCausalLM.from_pretrained("microsoft/biogpt", cache_dir=CACHEDIR)
  tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt", cache_dir=CACHEDIR)
  return tokenizer, model

#--
def read_BioGPT_large():
  model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT-Large", cache_dir=CACHEDIR)
  tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT-Large", cache_dir=CACHEDIR)
  return tokenizer, model

#--
if __name__ == "__main__" :

  #-
  with SLURMDistributedTorch(seed=SEED) as context:

    #- Benchmark LLM Runtimes
    benchmark_llm("BioGPT_base", read_BioGPT_base, LOADLIMIT, MAXLEN_BioGPT_base, DEBUGSET, context)
    benchmark_llm("BioGPT_large", read_BioGPT_large, LOADLIMIT, MAXLEN_BioGPT_large, DEBUGSET, context)