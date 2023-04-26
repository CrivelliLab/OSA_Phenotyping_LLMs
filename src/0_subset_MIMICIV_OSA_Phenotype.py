#- Imports
import os, argparse

#-
DEFAULT_MIMICIV_PATH = "/project/projectdirs/m1532/Projects_MVP/_datasets/MIMIC_IV/"
DEFAULT_OUTPATH = "output/records/"

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
  parser.add_argument("--pheno", type=str, description="Path to Dx classes.")
  parser.add_argument("--data", type=str, default=DEFAULT_MIMICIV_PATH, description="Path to MIMIC-IV.")
  parser.add_argument("--out", type=str, default=DEFAULT_OUTPATH, description="Path to output directory.")
  return parser.parse_args()

#--
if __name__ == "__main__":

  #- Parse cmd-line arguments and create logger
  args = parse_args()
  if not os.path.exists(args.out): mkpath(args.out)
  if not os.path.exists("logs/"): mkpath("logs/")
  logging.basicConfig(format="NLP@LBNL|%(asctime)s|%(name)s|%(levelname)s|%(message)s",
                        filename="logs/0_subset_MIMICIV_records.log",
                        level = logging.DEBUG)
  logger = logging.getLogger("__main__")

  #- Load Phenotype Dxs
  phenotypes = pd.read_csv(args.pheno)
  assert "ICD9" in phenotypes.columns
  assert "ICD10" in phenotypes.columns
  assert "CLASS" in phenotypes.columns
  logger.info("Phenotype dx loaded.")

  #- Load relavant MIMIC-IV tables
  # df = ...
  logger.info("MIMIC-IV tables subsetted.")

  #- Merge data and save to outpath
  outpath = "{}{}.parquet".format(args.outpath, ".".join(args.pheno.split("/")[-1].split(".")[:-1]))
  df.to_parquet(outpath, index=False)
  logger.info("Records stored under: {}".format(outpath))

  #- Report Stats Like Demographics, data volume, Dx rates, etc
  # logger.info("nb_unique_patients,{}".format(len(df.PID.unique())))




