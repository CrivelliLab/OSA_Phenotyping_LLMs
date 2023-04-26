#- Import Dependencies
import os, re, argparse, torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

#-
from nltk.tokenize.treebank import TreebankWordTokenizer
from transformers import AutoModel, AutoTokenizer
from mpi4py import MPI

#-- Preprocessing procedure used for NCBI_blueBERT
def NCBI_blueBert_preprocess(value):
  value = value.lower()
  value = re.sub(r'[\r\n]+', ' ', value)
  value = re.sub(r'[^\x00-\x7F]+', ' ', value)

  tokenized = TreebankWordTokenizer().tokenize(value)
  sentence = ' '.join(tokenized)
  sentence = re.sub(r"\s's\b", "'s", sentence)
  return sentence

#-- Measure the cosine distance between two NCBI_blueBERT hidden representations
#-- from the penultimate layer (last layer before logits).
def NCBI_blueBERT_cosine_distance(x1, tokens2, hidden2, model, tokenizer):

  #- Run model inference and collect penultimate states
  tokens1 = torch.Tensor(tokenizer(x1.to_list(), padding=True)["input_ids"]).long().cuda()
  hidden1 = model.forward(input_ids=tokens1, output_hidden_states=True, return_dict=True)["hidden_states"][-1]
  hidden1 = hidden1.sum(1, keepdim=True)

  #- Compute Cosine Distance
  norm1 = hidden1.norm(p=2, dim=2, keepdim=True)
  norm2 = hidden2.norm(p=2, dim=1, keepdim=True)
  norm = (norm1 * norm2.transpose(0,1)).clamp(1e-6)
  cosine = 1 - (torch.matmul(hidden1, hidden2.transpose(0,1))/norm)

  #-
  dists = []
  t1 = tokens1.cpu().detach().numpy()
  t2 = tokens2.cpu().detach().numpy()
  cosine = cosine.cpu().detach().numpy()
  for i in range(len(x1)):
    dists.append(cosine[i][0])
  return np.array(dists)

#--
def apply_NCBI_blueBERT_cosine_distance(search_string):
  model = AutoModel.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16").cuda()
  tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16")
  tokens2 = torch.Tensor(tokenizer(search_string)["input_ids"]).long().unsqueeze(0).cuda()
  hidden2 = model.forward(input_ids=tokens2, output_hidden_states=True, return_dict=True)["hidden_states"][-1][0]
  hidden2 = hidden2.sum(0, keepdim=True)
  apply = lambda x: NCBI_blueBERT_cosine_distance(x, tokens2, hidden2, model, tokenizer)
  return apply

#-- Command Line Interface
def parse_args():
  parser = argparse.ArgumentParser(description="Collect K nearest-neighbors to search string using NCBI_blueBERT language model.")
  parser.add_argument("--df", type=str, default="vocab.csv", help="Path to CSV containing search space.")
  parser.add_argument("--col", type=str, default="trigram", help="Text column used as search space.")
  parser.add_argument("--search", type=str, default="obstructive sleep apnea", help="Search string.")
  parser.add_argument("--batchsize", type=int, default=8, help="GPU batch size.")
  parser.add_argument("--outpath", type=str, default="knn.csv", help="Path to write nearest-neighbors output.")
  return parser.parse_args()

#-- MAIN
if __name__ == "__main__":

  #-
  args = parse_args()
  search_string = NCBI_blueBert_preprocess(args.search)

  #-
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  worldsize = comm.Get_size()
  print(rank,worldsize)

  #-
  print("Loading search space and preprocessing text...")
  df = pd.read_csv(args.df); tqdm.pandas()
  df = np.array_split(df, worldsize)[rank]
  search_space = df[args.col]
  search_space = search_space.progress_apply(NCBI_blueBert_preprocess)

  #-
  print("Calculating cosine distances...")
  compute = apply_NCBI_blueBERT_cosine_distance(search_string)
  chunks = np.array_split(search_space, len(search_space)//args.batchsize)
  dists = []
  for chunk in tqdm(chunks):
     dists.append(compute(chunk))

  #-
  cosine_distances = np.concatenate(dists)
  df["cosine"] = cosine_distances
  df = df[[args.col, "cosine"]].sort_values("cosine", ascending=True).reset_index(drop=True)
  df = df.reset_index().rename(columns={"index":"k"})

  #-
  print("Exporting nearest neighbor results...")
  df.to_csv("{}/part.{}.{}.csv".format(args.outpath, rank, worldsize), index=False)