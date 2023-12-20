#- Imports
import os, torch, logging, argparse, random
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpi4py import MPI

#-
from scipy.stats import entropy as entro
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression

#-
pt_size = 0.001
MIMIC4_REPORTS = "/global/cfs/cdirs/m1532/Projects_MVP/_datasets/MIMIC_IV/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv"
MIMIC4_ADMS = "/global/cfs/cdirs/m1532/Projects_MVP/_datasets/MIMIC_IV/physionet.org/files/mimiciv/2.2/hosp/admissions.csv"
MIMIC4_PATS = "/global/cfs/cdirs/m1532/Projects_MVP/_datasets/MIMIC_IV/physionet.org/files/mimiciv/2.2/hosp/patients.csv"
MIMIC4_DXS = "/global/cfs/cdirs/m1532/Projects_MVP/_datasets/MIMIC_IV/physionet.org/files/mimiciv/2.2/hosp/diagnoses_icd.csv"
# LLM_EMBEDS = "/global/cfs/cdirs/m1532/Projects_MVP/_models/embeds/LLMs/"
LLM_EMBEDS = "/pscratch/sd/r/rzamora/OSA_Phenotyping_LLMs/output/embeds/"
PHENOTYPES = "output/phenotype/"

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
def entropy(x, totals):
  """Allen R. Wilcox (1967). Indices of Qualitative Variation (No. ORNL-TM-1919). Oak Ridge National Lab., Tenn"""
  x = x.T
  totals = totals / totals.sum()
  e = torch.Tensor([entro(x[i]) for i in range(len(x))])
  e = torch.nan_to_num(e, 0.0)
  e = e @ totals
  return e.item() #/ np.log(len(totals))

#--
def logreg(x, y):
  model = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2')
  cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=666142)
  scores = cross_val_score(model, x, y, scoring='roc_auc_ovr_weighted', cv=cv, n_jobs=-1, verbose=True)
  return np.mean(scores), np.std(scores)

#--
def linreg(x, y):
  model = LinearRegression()
  cv = KFold(n_splits=10, random_state=666142, shuffle=True)
  scores = cross_val_score(model, x, y, cv=cv, n_jobs=-1, verbose=True, scoring='r2')
  return np.mean(scores), np.std(scores)

#--
def plot_umap(path, df, var, pt_size, cmap, transform=None, minv=None, maxv=None):
  per_cluster = df[["cluster", var]].groupby("cluster").mean().reset_index().rename(columns={var:"var_{}".format(var)})
  if "var_{}".format(var) in df.columns: df = df.drop(columns=["var_{}".format(var)])
  df = df.merge(per_cluster, on="cluster", how="left")
  if transform is not None: df["var_{}".format(var)] = df["var_{}".format(var)].apply(transform)
  plot_args = {"title": path.split("/")[-1][:-4],
               "x": "umap_1",
               "y": "umap_2",
               "c": "var_{}".format(var),
               "alpha": 1.0,
               "s": pt_size,
               "cmap":cmap}
  if minv is not None: plot_args["vmin"] = minv
  if maxv is not None: plot_args["vmax"] = maxv
  plt.rcParams['figure.figsize'] = [16, 16]
  ax = df[~df[var].isna()].plot.scatter(**plot_args)
  minax = min(df.umap_1.min(), df.umap_2.min()) - 0.5
  maxax = max(df.umap_1.max(), df.umap_2.max()) + 0.5
  ax.set_xlim(minax,maxax)
  ax.set_ylim(minax,maxax)
  ax.set_facecolor('black')
  plt.savefig(path, format="png", bbox_inches="tight")

#-- 
def map_phenos(x):
  if x == 3: return 2
  if x == 2: return 1
  if x == 1: return 3
  return 0

#--
def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--embed", type=str, default=LLM_EMBEDS)
  parser.add_argument("--labels", type=str)
  parser.add_argument("--out", type=str, default="output/entropy/")
  return parser.parse_args()

#--
if __name__ == "__main__":

  #-
  args = parse_args()
  OUTPATH = args.out
  logging.basicConfig(format="NLP@LBNL|%(asctime)s|%(name)s|%(levelname)s|%(message)s",
                      filename="logs/4_eval_MIMICIV_reports.log",
                      level = logging.DEBUG)
  logger = logging.getLogger("__main__")
  set_seed(666142)

  #-
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  worldsize = comm.Get_size()
  if rank == 0 and not os.path.exists("logs/"): mkpath("logs/")
  if rank == 0 and not os.path.exists(args.out): mkpath(args.out)

  #- Load MIMIC-IV Data
  pats = pd.read_csv(MIMIC4_PATS)
  pats["dob"] = pats.anchor_year - pats.anchor_age
  pats = pats[["subject_id", "gender", "dob", "dod"]]
  adms = pd.read_csv(MIMIC4_ADMS)
  adms = adms[["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "admission_type", "race"]]
  dxs = pd.read_csv(MIMIC4_DXS)
  dxs = dxs.drop_duplicates(subset=["subject_id","hadm_id","seq_num"]).groupby(["subject_id", "hadm_id"]).icd_code.count().reset_index()
  reports = pd.read_csv(MIMIC4_REPORTS)

  #- Merge MIMIC-IV Data
  df = adms.merge(pats, on="subject_id", how="left").reset_index(drop=True)
  df["deathtime"] = pd.to_datetime(df.deathtime)
  df["dod"] = pd.to_datetime(df.dod)
  df["death"] = df[["deathtime","dod"]].max(axis=1)
  df["dob"] = pd.to_datetime(df.dob.astype(str))
  df["admittime"] = pd.to_datetime(df.admittime)
  df["dischtime"] = pd.to_datetime(df.dischtime)
  df["lifespan"] = (df.dod - df.dob).dt.days
  df["from_birth"] = (df.dischtime - df.dob).dt.days
  df["till_death"] = ((df.dod - df.admittime) / np.timedelta64(1, 'h')) * 60
  df = df[["subject_id", "hadm_id", "admission_type", "gender", "race", "lifespan", "death", "from_birth", "till_death"]]
  df = df.merge(dxs, on=["subject_id", "hadm_id"], how="left")
  df = reports.merge(df, on=["subject_id", "hadm_id"], how="left")
  df = df[["subject_id", "hadm_id", "note_id", "storetime", "text", "admission_type", "gender", "race", "lifespan", "death", "from_birth", "till_death", "icd_code"]]
  del(pats); del(adms); del(dxs); del(reports)

  #- Medical Variables
  df["dead"] = 1 - df.death.isna().astype("int") # 1 is dead, 0 is alive
  df["age"] = df.from_birth // (365.25*5) # binned by 5-year intervals; not enough resolution for days.
  df["sex"] = (2*(df.gender == "M").astype("int")) + (df.gender == "F").astype("int") + 0 # 0 Missing, 1 is Female, 2 is Male ;
  # df["race"] = df.race.str.contains("WHITE") + ~df.race.str.contains("BLACK") + (~df.race.isna()).astype("int") # 0 Missing, 1 Black, 2 Other, 3 White
  df["race"] = (2*df.race.str.contains("WHITE")) + df.race.str.contains("BLACK").astype("int") + (~df.race.isna()).astype("int") # 0 Missing, 2 Black, 1 Other, 3 White
  df["race"] = df.race.apply(map_phenos)
  df["minstoDeathlog10"] = np.log10(np.clip(df.till_death, a_min=0, a_max=(60*24*365.25))) // 1.0 # Minutes remaining at time of admission, binned along log10
  df["dxslog2"] = np.log2(df.icd_code) // 1.0 # Number of dxs per admission, binned by log 2
  df["emergency"] = 1 - ((df.admission_type == "ELECTIVE") | (df.admission_type == "SURGICAL SAME DAY ADMISSION")).astype("int")
  
  #- 
  EMBEDS_PATH = args.embed
  models = list(sorted(os.listdir(EMBEDS_PATH)))
  models = [["{}{}/kmeans/{}".format(EMBEDS_PATH, m, f) for f in os.listdir("{}{}/kmeans/".format(EMBEDS_PATH, m))] for m in models]
  models = list(sorted([item for sublist in models for item in sublist]))
  models = np.array_split(models, worldsize)[rank]
  results = []
  for model in tqdm(models):

    #-  Load LLM Kmeans and UMAP
    llm = model.split("/")[-3].split(".")[0]
    layer = model.split("/")[-3].split(".")[3]
    kmeans = torch.load(model)
    umap = pd.read_csv(model.replace("/kmeans/", "/umap/").replace(".pt", ".nn.256.min.0.001.csv"))
    umap = umap.rename(columns={"id":"note_id"}).sort_values("sha224").reset_index(drop=True)
    latent = df.merge(umap, on="note_id", how="inner").reset_index(drop=True).reset_index().rename(columns={"index":"idx"})
    totals = F.one_hot(kmeans.argmin(-1)).sum(0).float() 
    clusters = F.one_hot(kmeans.argmin(-1), num_classes=len(totals))
    latent = latent.sort_values("sha224").reset_index(drop=True)
    latent["cluster"] = kmeans.argmin(-1).numpy()
    path = "{}{}.discharge.layer.{}.nclusters.{}.nn.256.min.0.001.".format(OUTPATH, llm, layer, len(totals))
    path += "var.{}.png"

    #- Sex Test
    col = "sex"
    male = clusters[latent.sex==2].sum(0).float() / totals
    female = clusters[latent.sex==1].sum(0).float() / totals
    e = entropy(torch.stack([male, female]), totals)
    plot_umap(path.format(col), latent, col, pt_size, "RdBu_r", transform=(lambda x: x-1), minv=0, maxv=1)
    # auc_mean, auc_std = logreg(clusters.numpy(), latent.sex)
    # results.append([llm, layer, len(totals), col, e, auc_mean, auc_std, None, None])
    results.append([llm, layer, len(totals), col, e, None, None, None, None])
    
    #- Race Test 
    col = "race"
    white = clusters[latent.race==3].sum(0).float() / totals
    other = clusters[latent.race==2].sum(0).float() / totals
    black = clusters[latent.race==1].sum(0).float() / totals
    e = entropy(torch.stack([white, black, other]), totals)
    plot_umap(path.format(col), latent, col, pt_size, "RdBu_r", transform=(lambda x: x-1))
    # auc_mean, auc_std = logreg(clusters.numpy(), latent.race)
    # results.append([llm, layer, len(totals), col, e, auc_mean, auc_std, None, None])
    results.append([llm, layer, len(totals), col, e, None, None, None, None])

    #- Age Test
    test = []
    col = "age"
    for nme, grp in latent.groupby(col):
        if np.isnan(nme): continue
        if nme < 4: continue
        if nme > 14: continue
        subset = clusters[grp.idx.to_numpy()].sum(0).float() 
        other = totals - subset
        e = entropy(torch.stack([subset/totals, other/totals]), totals)
        test.append([nme, e])
    test = pd.DataFrame(test, columns=[col, "entropy"]).sort_values(col).reset_index(drop=True)
    plot_umap(path.format(col), latent, col, pt_size, "cubehelix", transform=(lambda x: x*5), minv=20, maxv=100)
    # r2_mean, r2_std = linreg(np.expand_dims(test.age.to_numpy(), -1), np.expand_dims(test.entropy.to_numpy(), -1))
    # results.append([llm, layer, len(totals), col, test.entropy.sum(), None, None, r2_mean, r2_std])
    results.append([llm, layer, len(totals), col, test.entropy.sum(), None, None, None, None])

    #- Dxs Test
    test = []
    col = "dxslog2"
    for nme, grp in latent.groupby(col):
        if np.isnan(nme): continue
        if nme < 0: continue
        subset = clusters[grp.idx.to_numpy()].sum(0).float() 
        other = totals - subset
        e = entropy(torch.stack([subset/totals, other/totals]), totals)
        test.append([nme, e])
    test = pd.DataFrame(test, columns=[col, "entropy"]).sort_values(col).reset_index(drop=True)
    plot_umap(path.format(col), latent, col, pt_size, "cubehelix", minv=1, maxv=5)
    # r2_mean, r2_std = linreg(np.expand_dims(test.age.to_numpy(), -1), np.expand_dims(test.entropy.to_numpy(), -1))
    # results.append([llm, layer, len(totals), col, test.entropy.sum(), None, None, r2_mean, r2_std])
    results.append([llm, layer, len(totals), col, test.entropy.sum(), None, None, None, None])

    #- Admission Type Test
    col = "emergency"
    elective = clusters[latent.emergency==1].sum(0).float() / totals
    emergency = clusters[latent.emergency==0].sum(0).float() / totals
    e = entropy(torch.stack([elective, emergency]), totals)
    plot_umap(path.format(col), latent, col, pt_size, "RdBu_r", minv=0, maxv=1)
    # auc_mean, auc_std = logreg(clusters.numpy(), latent.dead)
    # results.append([llm, layer, len(totals), col, e, auc_mean, auc_std, None, None])
    results.append([llm, layer, len(totals), col, e, None, None, None, None])

    #- Death Test 
    col = "dead"
    dead = clusters[latent.dead==1].sum(0).float() / totals
    alive = clusters[latent.dead==0].sum(0).float() / totals
    e = entropy(torch.stack([dead, alive]), totals)
    plot_umap(path.format(col), latent, col, pt_size, "RdBu_r", minv=0, maxv=1)
    # auc_mean, auc_std = logreg(clusters.numpy(), latent.dead)
    # results.append([llm, layer, len(totals), col, e, auc_mean, auc_std, None, None])
    results.append([llm, layer, len(totals), col, e, None, None, None, None])

    #- Life Test
    test = []
    col = "minstoDeathlog10"
    for nme, grp in latent.groupby(col):
        if np.isnan(nme): continue
        if nme < 0: continue
        if nme > 15: continue
        subset = clusters[grp.idx.to_numpy()].sum(0).float() 
        other = totals - subset
        e = entropy(torch.stack([subset/totals, other/totals]), totals)
        test.append([nme, e])
    test = pd.DataFrame(test, columns=[col, "entropy"]).sort_values(col).reset_index(drop=True)
    plot_umap(path.format(col), latent, col, 0.01, "cubehelix_r", minv=2, maxv=5)
    # r2_mean, r2_std = linreg(np.expand_dims(test.age.to_numpy(), -1), np.expand_dims(test.entropy.to_numpy(), -1))
    # results.append([llm, layer, len(totals), col, test.entropy.sum(), None, None, r2_mean, r2_std])
    results.append([llm, layer, len(totals), col, test.entropy.sum(), None, None, None, None])

    #- PhenoTypes
    phenos = sorted(["{}{}".format(PHENOTYPES, f) for f in os.listdir(PHENOTYPES) if not f.startswith("osa")])
    OSA = pd.read_csv("{}osa.csv".format(PHENOTYPES))
    for p in phenos:
      col = "osa.{}".format(p.split("/")[-1].split(".")[0])
      PHENO = pd.read_csv(p)
      latent["pheno"] = (2*latent.hadm_id.isin(OSA.hadm_id)) + latent.hadm_id.isin(PHENO.hadm_id) + 0
      latent["pheno"] = latent.pheno.apply(map_phenos)
      combo = clusters[latent.pheno==2].sum(0).float() / totals
      osa = clusters[latent.pheno==1].sum(0).float() / totals
      comorbid = clusters[latent.pheno==3].sum(0).float() / totals
      e = entropy(torch.stack([osa, combo, comorbid]), totals)
      plot_umap(path.format(col), latent[latent["pheno"]>0], "pheno", 0.01, "RdBu_r", transform=(lambda x: x-1), minv=0, maxv=2)
      # auc_mean, auc_std = logreg(clusters.numpy(), latent.race)
      # results.append([llm, layer, len(totals), col, e, auc_mean, auc_std, None, None])
      results.append([llm, layer, len(totals), col, e, None, None, None, None])

  #-
  cols = ["model", "layer", "nb_clusters", "variable", "entropy", "auc_mean", "auc_std", "r2_mean", "r2_std"]
  results = pd.DataFrame(results, columns=cols)

  #-
  results = comm.gather(results, root=0)
  if rank == 0:
   results = pd.concat(results)
   results.to_csv("{}entropy.csv".format(OUTPATH), index=False)
   print(results)