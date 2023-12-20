#-
from models.src.biobart import read_BioBART_base, read_BioBART_large, MAXLEN_BioBart_base, MAXLEN_BioBart_large
from models.src.biobert import read_BioBERT_base, read_BioBERT_large, MAXLEN_BioBert_base, MAXLEN_BioBert_large
from models.src.biogpt import read_BioGPT_base, read_BioGPT_large, MAXLEN_BioGPT_base, MAXLEN_BioGPT_large
from models.src.biomegatron import read_BioMegatron_base, MAXLEN_BioMegatron_base
from models.src.clinicalbert import read_Bio_ClinicalBERT, MAXLEN_Bio_ClinicalBERT
from models.src.clinicalt5 import read_ClinicalT5_base, read_ClinicalT5_large, MAXLEN_ClinicalT5_base, MAXLEN_ClinicalT5_large
from models.src.gatortron import read_Gatortron_base, read_Gatortron_s, read_Gatortron_medium, MAXLEN_Gatortron_base, MAXLEN_Gatortron_s, MAXLEN_Gatortron_medium
from models.src.radbert import read_RadBERT_2m, read_RadBERT_4m, MAXLEN_RadBERT_2m, MAXLEN_RadBERT_4m

#-
def read_llm(nme):
  if   nme == "BioBART_base": tokenizer, model = read_BioBART_base(); return tokenizer, model, MAXLEN_BioBart_base
  elif nme == "BioBART_large": tokenizer, model = read_BioBART_large(); return tokenizer, model, MAXLEN_BioBart_large
  elif nme == "BioBERT_base": tokenizer, model = read_BioBERT_base(); return tokenizer, model, MAXLEN_BioBert_base
  elif nme == "BioBERT_large": tokenizer, model = read_BioBERT_large(); return tokenizer, model, MAXLEN_BioBert_large
  elif nme == "BioGPT_base": tokenizer, model = read_BioGPT_base(); return tokenizer, model, MAXLEN_BioGPT_base
  elif nme == "BioGPT_large": tokenizer, model = read_BioGPT_large(); return tokenizer, model, MAXLEN_BioGPT_large
  elif nme == "BioMegatron_base": tokenizer, model = read_BioMegatron_base(); return tokenizer, model, MAXLEN_BioMegatron_base
  elif nme == "Bio_ClinicalBERT": tokenizer, model = read_Bio_ClinicalBERT(); return tokenizer, model, MAXLEN_Bio_ClinicalBERT
  elif nme == "ClinicalT5_base": tokenizer, model = read_ClinicalT5_base(); return tokenizer, model, MAXLEN_ClinicalT5_base
  elif nme == "ClinicalT5_large": tokenizer, model = read_ClinicalT5_large(); return tokenizer, model, MAXLEN_ClinicalT5_large
  elif nme == "Gatortron_base": tokenizer, model = read_Gatortron_base(); return tokenizer, model, MAXLEN_Gatortron_base
  elif nme == "Gatortron_s": tokenizer, model = read_Gatortron_s(); return tokenizer, model, MAXLEN_Gatortron_s
  elif nme == "Gatortron_medium": tokenizer, model = read_Gatortron_medium(); return tokenizer, model, MAXLEN_Gatortron_medium
  elif nme == "RadBERT_2m": tokenizer, model = read_RadBERT_2m(); return tokenizer, model, MAXLEN_RadBERT_2m
  elif nme == "RadBERT_4m": tokenizer, model = read_RadBERT_4m(); return tokenizer, model, MAXLEN_RadBERT_4m
  else: return None