#--
import os, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot

#-- Command Line Interface
def parse_args():
  parser = argparse.ArgumentParser(description="Trains and evaluates a Multinomial Logistic Regression model.")
  parser.add_argument("--topk", type=int, default=10000, help="Top K NN as vocab.")
  parser.add_argument("--c", type=float, default=0.1, help="Regularization strength.")
  parser.add_argument("--iters", type=int, default=500, help="Number of LR iterations.")
  parser.add_argument("--outpath", type=str, default="lr_model.csv", help="Path to write nearest-neighbors output.")
  return parser.parse_args()

#--
if __name__ == "__main__":

  #-
  args = parse_args()

  #- Load Dataset
  print("Loading OSA+HF dataset...")
  inputs = pd.read_csv("inputs.csv")
  labels = pd.read_csv("labels.csv")
  df = inputs.merge(labels, on=["SUBJECT_ID", "HADM_ID"], how="inner")
  df["LABEL"] = df[["hf", "hf_sa"]].apply(lambda x: int(x["hf"])+int(x["hf_sa"]), axis=1)
  X = df["TEXT"]
  Y = df["LABEL"]

  #- Preprocess Text into Bag-of-words
  print("Processing Text...")
  vocab = pd.read_csv("NNsearch_OSA.csv").head(args.topk)
  cvec = CountVectorizer(lowercase=True, stop_words="english", ngram_range=(3,3), vocabulary=vocab["trigrams"])
  X = cvec.fit_transform(X)
  X[X>1] = 1

  #- Train and Evaluate Model
  print("Training Multinomial Logistic Regression")
  model = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', C=args.c, max_iter=args.iters)
  cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=123456)
  scores = cross_val_score(model, X, Y, scoring='roc_auc_ovr_weighted', cv=cv, n_jobs=-1, verbose=True)
  print('KFold-Results: >%s %.3f (%.3f)' % (args.c, np.mean(scores), np.std(scores)))

  #- Fit single model
  for train_index, test_index in cv.split(X, Y):
    model=model.fit(X[train_index], Y[train_index])
    break
  print(classification_report(Y, model.predict(X)))
  print(confusion_matrix(Y, model.predict(X)))

  #- Export Model Coefficients
  coefs = vocab[["trigrams", "cosine"]]
  coefs["osa_coef"] = model.coef_[0]
  coefs["hf_coef"] = model.coef_[1]
  coefs["hf_osa_coef"] = model.coef_[2]
  coefs["osa_intercept"] = model.intercept_[0]
  coefs["hf_intercept"] = model.intercept_[1]
  coefs["hf_osa_intercept"] = model.intercept_[2]
  coefs["roc_auc_ovr_weighted_mean"] = np.mean(scores)
  coefs["roc_auc_ovr_weighted_std"] = np.std(scores)
  ceofs = coefs.sort_values("osa_coef", ascending=False)
  ceofs.to_csv(args.outpath, index=False)