#-
from transformers import BertTokenizer, AutoModel
from utils.parallel import SLURMDistributedTorch
from utils.test import benchmark_llm

#-
SEED = 666142
CACHEDIR = "/global/cfs/cdirs/m1532/Projects_MVP/_models/LLMs/huggingface_cache/"
MAXLEN_BioMegatron_base = 512 #- from: https://huggingface.co/EMBO/BioMegatron345mUncased/blob/main/config.json
LOADLIMIT = 128
DEBUGSET = ["Patient had a heart attack on their way to work.", 
            "Patient had a heart attack on their way to work."]

#--
def read_BioMegatron_base():
  tokenizer = BertTokenizer.from_pretrained("EMBO/BioMegatron345mCased", cache_dir=CACHEDIR)
  model = AutoModel.from_pretrained("EMBO/BioMegatron345mCased", cache_dir=CACHEDIR)
  return tokenizer, model

#--
if __name__ == "__main__" :

  #-
  with SLURMDistributedTorch(seed=SEED) as context:

    #- Benchmark LLM Runtimes
    benchmark_llm("BioMegatron_base", read_BioMegatron_base, LOADLIMIT, MAXLEN_BioMegatron_base, DEBUGSET, context)