#- Imports
import os, argparse, torch, random, logging
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

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
def read_LLM_embeddings(path):
  shas = []
  embeds = []
  for f in os.listdir("{}embeds/".format(path)):
    data = torch.load("{}embeds/{}".format(path, f))
    shas += data["sha224"]
    embeds.append(data["latent"])
  embeds = torch.cat(embeds)
  df = pd.DataFrame({"sha224": shas})
  return df, embeds

#--
def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--embeds", type=str)
  parser.add_argument("--nclusters", type=int)
  return parser.parse_args()

#--
if __name__ == "__main__":

  #- Parse cmd-line arguments and create logger
  args = parse_args()
  if not os.path.exists("logs/"): mkpath("logs/")
  logging.basicConfig(format="NLP@LBNL|%(asctime)s|%(name)s|%(levelname)s|%(message)s",
                      filename="logs/2_clustr_MIMICIV_reports.log",
                      level = logging.DEBUG)
  logger = logging.getLogger("__main__")
  set_seed(666142)

  #-
  n_clusters = args.nclusters

  #- Load Embeddings
  EMBEDPATH = args.embeds
  df, latent = read_LLM_embeddings(EMBEDPATH)
  latent = F.normalize(latent, dim=0)
  df = df.reset_index(drop=True).reset_index().rename(columns={"index": "idx"})

  #- 
  kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init="auto", random_state=666142, verbose=1, max_no_improvement=250).fit(latent)
  clusters = torch.Tensor(kmeans.transform(latent))

  #- Join with Doc Ids
  ids = pd.concat([pd.read_csv("{}ids/{}".format(EMBEDPATH, f)) for f in os.listdir("{}ids/".format(EMBEDPATH))])
  ids = ids.merge(df, on="sha224", how="left").sort_values("sha224").reset_index(drop=True)

  clusters = torch.cat([clusters[int(i)].unsqueeze(0) for i in ids.idx])

  #- Store to File.
  label = "kmeans"
  if not os.path.exists("{}{}/".format(EMBEDPATH,label)): mkpath("{}{}/".format(EMBEDPATH,label))
  torch.save(clusters, "{}{}/nclusters.{}.pt".format(EMBEDPATH, label, n_clusters))

