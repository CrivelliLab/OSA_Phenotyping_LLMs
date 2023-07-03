#- Imports
import math
import os, argparse
import pandas as pd
from hashlib import sha224
import logging
import argparse

#-
DEFAULT_MIMICIV_PATH = "/project/projectdirs/m1532/Projects_MVP/_datasets/MIMIC_IV/"
DEFAULT_OUTPATH = "/output/phenotype/"

#--
def mkpath(path):
  path = path if path.endswith("/") else path+"/"
  nested = path.split("/")
  curr = nested[0]
  for n in nested[1:]:
    if not os.path.exists(curr): os.mkdir(curr)
    curr += "/" + n
  return path

#--
def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--pheno", type=str)
  parser.add_argument("--data", type=str, default=DEFAULT_MIMICIV_PATH)
  parser.add_argument("--out", type=str, default=DEFAULT_OUTPATH)
  return parser.parse_args()

#-
def subset_patients(df, path_to_MIMIC):
    
    #-
    patients = pd.read_csv("{}mimiciv/2.2/hosp/patients.csv".format(path_to_MIMIC))
    longtitle_diagnoses = pd.read_csv("{}mimiciv/2.2/hosp/d_icd_diagnoses.csv".format(path_to_MIMIC))
    matching_diagnoses = pd.read_csv("{}mimiciv/2.2/hosp/diagnoses_icd.csv".format(path_to_MIMIC))

    #-
    df["ICD9"] = df["ICD9"].str.replace('"', '').str.replace('.', '')
    df["ICD10"] = df["ICD10"].str.replace('"', '').str.replace('.', '')
    icd9_values = df["ICD9"].dropna().tolist()
    icd10_values = df["ICD10"].dropna().tolist()
    combined_values = icd9_values + icd10_values
    comorbidity_df = matching_diagnoses[matching_diagnoses["icd_code"].isin(combined_values)]
    comorbidity_string = longtitle_diagnoses[longtitle_diagnoses["icd_code"].isin(combined_values)]
    comorbidity_codeandtitle = comorbidity_df.merge(comorbidity_string, on = "icd_code", how = "inner")
    comorbidity_pheno = comorbidity_codeandtitle[["subject_id","hadm_id","seq_num", "icd_code","icd_version_x","long_title"]]
    comorbidity_pheno.rename(columns={'icd_version_x': 'icd_version'})
    return comorbidity_pheno

#--
if __name__ == "__main__":

  #-
  args = parse_args()
  if not os.path.exists(args.out): mkpath(args.out)
  if not os.path.exists("logs/"): mkpath("logs/")
  logging.basicConfig(format="NLP@LBNL|%(asctime)s|%(name)s|%(levelname)s|%(message)s",
                        filename="logs/0_subset_MIMICIV_phenotype.log",
                        level = logging.DEBUG)
  logger = logging.getLogger("__main__")
  
  #-
  dxs = pd.read_csv(args.pheno)
  phenotype = subset_patients(dxs, args.data)
  phenotype["phenotype"] = args.pheno.split("/")[-1].split(".")[0]

  #-
  nb_patients = len(phenotype.subject_id.unique())
  nb_longtitle = len(phenotype.long_title.unique())
  nb_icdcode = len(phenotype.icd_code.unique())
  nb_uniquesubjectvisitcodeversiontitle = len(phenotype.hadm_id.unique())
  logger.info("Number of unique patients: {}".format(nb_patients))
  logger.info("Number of unique long titles: {}".format(nb_longtitle))
  logger.info("Number of unique icd codes: {}".format(nb_icdcode))
  logger.info("Number of unique hadm: {}".format(nb_uniquesubjectvisitcodeversiontitle))

  #-
  outpath = "{}/{}.csv".format(args.out, ".".join(args.pheno.split("/")[-1].split(".")[:-1])) 
  phenotype.to_csv(outpath, index=False)
  logger.info("Records stored under: {}".format(outpath))