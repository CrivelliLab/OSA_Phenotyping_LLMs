#- Imports
import os, argparse, torch, random
import numpy as np
import pandas as pd

#-
from umap import UMAP

#-
DEFAULT_OUTPATH = "output/UMAP/"

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
  parser.add_argument("--records", type=str, description="Path to subsetted records.")
  parser.add_argument("--out", type=str, default=DEFAULT_OUTPATH, description="Path to output directory.")
  return parser.parse_args()


#--
if __name__ == "__main__":

  #- Parse cmd-line arguments and create logger
  args = parse_args()
  if not os.path.exists(args.out): mkpath(args.out)
  if not os.path.exists("logs/"): mkpath("logs/")
  logging.basicConfig(format="NLP@LBNL|%(asctime)s|%(name)s|%(levelname)s|%(message)s",
                        filename="logs/2_umap2D_MIMICIV_latent.log",
                        level = logging.DEBUG)
  logger = logging.getLogger("__main__")
  set_seed(666142)

  #- Load Records
  df = pd.read_parquet(args.records)[["sha224"]]

  #- Load Latent Space
  record_nm = ".".join(args.records.split("/")[-1].split(".")[:-1])
  path = "{}{}/".format(args.outpath, record_nm)
  latent = torch.cat([torch.load("{}{}".format(path,f)) for f in os.listdir(path)])
  assert len(latent) == len(df)

  #- Apply UMAP
  #*** This will have to be calibrated and scaled to corpus.
  transform = UMAP(n_neighbors=100, min_dist=0.1, n_components=2, metric="cosine", random_state=666142)
  umapped = transform.fit_transform(latent)
  umapped = pd.DataFrame(X_embed, columns=["umap_0", "umap_1"])
  umapped["sha224"] = df.sha224

  #- Store to File.
  umapped.to_parquet(arg.records.replace("/records/", "/UMAP/"), index=False)
  

