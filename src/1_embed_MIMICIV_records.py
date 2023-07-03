#- Imports
import os, argparse, torch, random, logging
import numpy as np
import pandas as pd
from hashlib import sha224
from tqdm import tqdm
from models.src.api import read_llm
from models.src.utils.parallel import SLURMDistributedTorch
from models.src.utils.embed import tokenize_texts, embed_tokens_tensor

#- NOTES
# * 331,793 unique discharge reports
# * 2,299,451 unique radiology reports

#-
SEED = 666142
DEFAULT_OUTPATH = "output/embeds/"
MIMICIV_COLS = {"note_id":"id", "text":"text"}

#--
def read_LLM_embeddings(path):
  ids = pd.concat([pd.read_csv("{}ids/{}".format(path, f)) for f in os.listdir("{}ids/")])
  shas = []
  embeds = []
  for f in os.listdir("{}embeds/"):
    data = torch.load("{}embeds/{}".format(path, f))
    shas += data["sha224"]
    embeds.append(data["latent"])
  embeds = torch.cat(embeds)
  df = pd.DataFrame({"sha224": shas, "latent": embeds})
  df = ids.merge(df, on="sha224", how="left")
  return df

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
  parser.add_argument("--data", type=str, help="Path to text corpus table.")
  parser.add_argument("--model", type=str, help="Model name.")
  parser.add_argument("--max_batch", type=int, default=8, help="GPU batch size.")
  parser.add_argument("--layer", type=int, default=0, help="Layer index from tail.")
  parser.add_argument("--out", type=str, default=DEFAULT_OUTPATH, help="Path to output directory.")
  return parser.parse_args()

#--
if __name__ == "__main__":

  #-
  with SLURMDistributedTorch(seed=SEED) as context:
    
    #- Parse cmd-line arguments and create logger
    args = parse_args()
    logging.basicConfig(format="NLP@LBNL|%(asctime)s|%(name)s|%(levelname)s|%(message)s",
                          filename="logs/1_embed_MIMICIV_records.log",
                          level = logging.DEBUG)
    logger = logging.getLogger("__main__")
    OUTPATH = "{}{}.{}.layer.{}/".format(args.out, args.model, args.data.split("/")[-1].split(".")[0], args.layer)

    #- Create Directories
    if context.rank == 0:
      if not os.path.exists("logs/"): mkpath("logs/")
      if not os.path.exists("{}ids/".format(OUTPATH)): mkpath("{}ids/".format(OUTPATH))
      if not os.path.exists("{}toks/".format(OUTPATH)): mkpath("{}toks/".format(OUTPATH))
      if not os.path.exists("{}embeds/".format(OUTPATH)): mkpath("{}embeds/".format(OUTPATH))
    context.barrier()

    #- Load Model
    max_batch = args.max_batch
    tokenizer, model, max_seq = read_llm(args.model)
    if getattr(tokenizer, "token_to_id", None):
      pad = tokenizer.token_to_id("<PAD>")
      pad = tokenizer.token_to_id("<pad>") if pad is None else pad
      pad = tokenizer.token_to_id("[PAD]") if pad is None else pad
    else:
      pad = tokenizer.convert_tokens_to_ids("<PAD>")
      pad = tokenizer.convert_tokens_to_ids("<pad>") if pad is None else pad
      pad = tokenizer.convert_tokens_to_ids("[PAD]") if pad is None else pad
    assert pad is not None
    model.to(context.device)
    context.barrier()

    #- Partition Corpus
    IDPATH = "{}ids/part.{}.{}.csv".format(OUTPATH, context.rank, context.world_size)
    if not os.path.exists(IDPATH):
      df = pd.read_csv(args.data)[MIMICIV_COLS.keys()].rename(columns=MIMICIV_COLS).sort_values("id")
      df = np.array_split(df, context.world_size)[context.rank]
      df["sha224"] = df.text.apply(lambda x: sha224(x.encode("utf-8")).hexdigest())
      df = df.sort_values("sha224").reset_index(drop=True)
      df[["id", "sha224"]].to_csv(IDPATH, index=False)
    else:
      df = pd.read_csv(args.data)[MIMICIV_COLS.keys()].rename(columns=MIMICIV_COLS).sort_values("id")[["id", "text"]]
      df = pd.read_csv(IDPATH).merge(df, on="id", how="left")
    context.barrier()  

    #- Tokenize Partition
    TOKPATH = "{}toks/part.{}.{}.csv".format(OUTPATH, context.rank, context.world_size)
    if not os.path.exists(TOKPATH):
      df = df[["sha224", "text"]].drop_duplicates("sha224").reset_index(drop=True)
      df["toks"] = tokenize_texts(df.text, tokenizer)
      df["nb_toks"] = df.toks.apply(len)
      toks_split = []
      for i, row in df.iterrows():
        sha, toks, nb_toks = row[["sha224", "toks", "nb_toks"]]
        splits = np.array_split(toks, np.ceil(nb_toks/max_seq))
        # if nb_toks > (max_seq): splits += np.array_split(toks[max_seq//2:], np.ceil((nb_toks - (max_seq//2)) // max_seq))
        for split in splits:
          toks_split.append((sha, [split[s] if s < len(split) else pad for s in range(max_seq)], len(split)))
      df = pd.DataFrame(toks_split, columns=["sha224", "toks", "nb_toks"])
      df["toks"] = df.toks.apply(str)
      df.to_csv(TOKPATH, index=False)
    else: df = pd.read_csv(TOKPATH)
    context.barrier()

    #- Embed Partitions
    EMBEDPATH = "{}embeds/part.{}.{}.pt".format(OUTPATH, context.rank, context.world_size)
    if not os.path.exists(EMBEDPATH):
      latent = []
      dfs = np.array_split(df, np.ceil(len(df)/max_batch))
      for d in tqdm(dfs):
        tokens = d["toks"].apply(lambda x: [int(xx) for xx in x[1:-1].split(", ")]).tolist()
        tokens = torch.Tensor(tokens).long().to(context.device)
        lens = d.nb_toks.tolist() #.unsqueeze(1).to(context.device)
        embeds = embed_tokens_tensor(tokens, model, layer_id=(-1-args.layer), agg=None)
        embeds = embeds.cpu()
        embeds = torch.cat([embeds[i:(i+1), 0:lens[i]].mean(1) for i in range(len(d))])
        latent.append(embeds.detach().clone())
        del(tokens); del(embeds)
        torch.cuda.empty_cache()
      latent = torch.cat(latent)
      df = pd.concat(dfs).reset_index(drop=True).reset_index()
      latent_ = []
      signs = []
      for sha, grp in tqdm(df.groupby("sha224"), total=len(df.sha224.unique())):
        ids_ = grp["index"].tolist()
        embed = torch.cat([latent[i].unsqueeze(0) for i in grp["index"].tolist()]).mean(0).unsqueeze(0)
        latent_.append(embed)
        signs.append(sha)
      latent = torch.cat(latent_)
      data = {"sha224":signs, "latent":latent}
      torch.save(data, EMBEDPATH)
