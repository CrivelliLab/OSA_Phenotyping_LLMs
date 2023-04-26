#- Imports
import os, argparse, torch, random
import numpy as np
import pandas as pd
from mpi4py import MPI

#-
DEFAULT_OUTPATH = "output/X/"

#-
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

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
  parser.add_argument("--model", type=str, description="Model path.")
  parser.add_argument("--records", type=str, description="Path to subsetted records.")
  parser.add_argument("--batchsize", type=int, default=8, help="GPU batch size.")
  parser.add_argument("--out", type=str, default=DEFAULT_OUTPATH, description="Path to output directory.")
  return parser.parse_args()

#-- *** This needs to be implemented
def load_LLM_latent(model_path):
  return None

#--
if __name__ == "__main__":

  #- Parse cmd-line arguments and create logger
  args = parse_args()
  if not os.path.exists("logs/"): mkpath("logs/")
  logging.basicConfig(format="NLP@LBNL|%(asctime)s|%(name)s|%(levelname)s|%(message)s",
                        filename="logs/0_subset_MIMICIV_records.log",
                        level = logging.DEBUG)
  logger = logging.getLogger("__main__")

  #- MPI Context
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  worldsize = comm.Get_size()

  #- Load Reports Per Worker
  df = pd.read_parquet(args.reports)
  df = np.array_split(df, worldsize)[rank]
  assert "text" in df.columns

  #- Load Model
  model = load_LLM_latent(args.model)

  #- Run inference
  latent = []
  text_batches = np.array_split(df.text, args.batchsize)
  for batch in text_batches:
    embeddings = model(batch)
    latent.append(embeddings)
  latent = torch.cat(latent)

  #- Store Embedding Tensors
  report_nm = ".".join(args.reports.split("/")[-1].split(".")[:-1])
  outpath = "{}{}/".format(args.outpath, report_nm)
  if not os.path.exists(outpath): mkpath(outpath)
  outpath = "{}part.{}.{}.pt".format(outpath, rank, world_size)
  torch.save(latent, outpath)
  
