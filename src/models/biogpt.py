#-
import os, torch
from fairseq.models.transformer_lm import TransformerLanguageModel

#-
MODELPATH = "/global/cfs/cdirs/m1532/Projects_MVP/_models/LLMs/Pre-trained-BioGPT/"

#--
def load_BioGPT_Base(model_path=MODELPATH, device="cpu"):
  assert os.path.exists(model_path)
  m = TransformerLanguageModel.from_pretrained(
          model_path, 
          "checkpoint.pt", 
          "data",
          tokenizer='moses', 
          bpe='fastbpe', 
          bpe_codes="data/bpecodes",
          min_len=100,
  m.to(device)
  return m
  
