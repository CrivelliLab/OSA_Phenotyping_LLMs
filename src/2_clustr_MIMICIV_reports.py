#- Imports
import os, argparse, torch, random, logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from hdbscan import HDBSCAN
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
  n_neighbors = 512
  min_dist= 0.8

  #- Load Embeddings
  EMBEDPATH = args.embeds
  df, latent = read_LLM_embeddings(EMBEDPATH)

  #- Apply UMAP
  # #*** This will have to be calibrated and scaled to corpus.
  transform = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric="correlation", random_state=666142)
  umapped = transform.fit_transform(latent)
  df["umap_1"] = umapped[:,0]
  df["umap_2"] = umapped[:,1]

  #- Join with Doc Ids
  ids = pd.concat([pd.read_csv("{}ids/{}".format(EMBEDPATH, f)) for f in os.listdir("{}ids/".format(EMBEDPATH))])
  ids = ids.merge(df, on="sha224", how="left")

  #- Store to File.
  if not os.path.exists("{}umap/".format(EMBEDPATH)): mkpath("{}umap/".format(EMBEDPATH))
  ids.to_csv("{}umap/nn.{}.min.{}.csv".format(EMBEDPATH, n_neighbors, min_dist), index=False)

  #-
  idxs = transform._knn_indices
  dists = transform._knn_dists
  row = []
  col = []
  data = []
  for i in range(idxs.shape[0]):
    for j in range(idxs.shape[1]):
      row.append(i)
      col.append(j)
      data.append(dists[i,j])
  dists = csr_array((data, (row, col)), shape=(len(idxs), len(idxs)))

  #- HDBSCAN Unsupervised Clustering
  clusterer = HDBSCAN(core_dist_n_jobs=16, metric="precomputed")
  clusterer.fit(dists)
  df["cluster"] = clusterer.labels_
  df["probs"] = clusterer.probabilities_

  #- Join with Doc Ids
  ids = pd.concat([pd.read_csv("{}ids/{}".format(EMBEDPATH, f)) for f in os.listdir("{}ids/".format(EMBEDPATH))])
  ids = ids.merge(df, on="sha224", how="left")

  #- Store to File.
  if not os.path.exists("{}hdbscan/".format(EMBEDPATH)): mkpath("{}hdbscan/".format(EMBEDPATH))
  ids.to_csv("{}hdbscan/nn.{}.min.{}.csv".format(EMBEDPATH, n_neigbors, min_dist), index=False)
