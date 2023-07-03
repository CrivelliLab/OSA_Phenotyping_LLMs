#- Imports
import os, argparse, torch, random, logging
import numpy as np
import pandas as pd
from umap import UMAP

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
def read_LLM_embeddings(path, n_clusters):
  shas = []
  for f in os.listdir("{}embeds/".format(path)):
    data = torch.load("{}embeds/{}".format(path, f))
    shas += data["sha224"]
  embeds = torch.load("{}kmeans/nclusters.{}.pt".format(path, n_clusters))
  embeds = embeds / embeds.max()
  df = pd.DataFrame({"sha224": shas})
  return df, embeds

#--
def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--embeds", type=str)
  parser.add_argument("--nclusters", type=int)
  parser.add_argument("--nneighbors", type=int)
  parser.add_argument("--mindist", type=float)
  return parser.parse_args()

#--
if __name__ == "__main__":

  #- Parse cmd-line arguments and create logger
  args = parse_args()
  if not os.path.exists("logs/"): mkpath("logs/")
  logging.basicConfig(format="NLP@LBNL|%(asctime)s|%(name)s|%(levelname)s|%(message)s",
                      filename="logs/2_umap2D_MIMICIV_latent.log",
                      level = logging.DEBUG)
  logger = logging.getLogger("__main__")
  set_seed(666142)

  #-
  n_clusters = args.nclusters
  n_neighbors = args.nneighbors
  min_dist = args.mindist

  #- Load Embeddings
  EMBEDPATH = args.embeds
  df, latent = read_LLM_embeddings(EMBEDPATH, n_clusters)

  #- Join with Doc Ids
  ids = pd.concat([pd.read_csv("{}ids/{}".format(EMBEDPATH, f)) for f in os.listdir("{}ids/".format(EMBEDPATH))])
  ids = ids.merge(df, on="sha224", how="left").sort_values("sha224").reset_index(drop=True)
  df = ids

  #- Apply UMAP
  # #*** This will have to be calibrated and scaled to corpus.
  transform = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric="euclidean")
  umapped = transform.fit_transform(latent)
  df["umap_1"] = umapped[:,0]
  df["umap_2"] = umapped[:,1]

  #- Store to File.
  if not os.path.exists("{}umap/".format(EMBEDPATH)): mkpath("{}umap/".format(EMBEDPATH))
  df.to_csv("{}umap/nclusters.{}.nn.{}.min.{}.csv".format(EMBEDPATH, n_clusters, n_neighbors, min_dist), index=False)

