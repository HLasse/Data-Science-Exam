## Models not using the encoded representation
import os
from pathlib import Path
import logging
import argparse
import pickle

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.linear_model import  LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def setwd(path="GenomeDK_genbert"):
    home = Path.home()
    wd = os.path.abspath(os.path.join(home, 'NLPPred', path))
    os.chdir(wd)


def create_models():
    model_dict = {"SVM rbf" : svm.SVC(random_state=42),
                  "SVM linear" : svm.SVC(kernel = "linear", random_state=42),
                  "Logistic Regression" : LogisticRegression(random_state=42),
                  "Lasso" : SGDClassifier(loss="log", penalty="l1", random_state=42),
                  "Elastic net" : SGDClassifier(loss="log", penalty="elasticnet"),
                  "Random Forest" : RandomForestClassifier(random_state=42)}
   
    names = []
    model_list = []
    for name, model in model_dict.items():
        model_list.append(Pipeline([('clf', model)]))
        model_list.append(Pipeline([('pca', TruncatedSVD(n_components=128)),
                                      ('clf', model)]))
        model_list.append(Pipeline([('pca', TruncatedSVD(n_components=233)),
                                      ('clf', model)]))
        model_list.append(Pipeline([('pca', TruncatedSVD(n_components=699)),
                                ('clf', model)]))
         
        names.extend([name, name + "SVD 128", name + "SVD 233", name + "SVD 699"])
    
    return dict(zip(names, model_list))




def main():
    setwd("inter_data")
    with open("x_test_1h.dat", "rb") as f:
        X = pickle.load(f)
    with open("y_test.npy", 'rb') as f:
        y = np.load(f)
    with open("x_val_1h.dat", 'rb') as f:
        X_v = pickle.load(f)
    with open("y_val.npy", 'rb') as f:
        y_v = np.load(f)

    models = create_models()

    accuracy, prec, rec, auc, f1 = [], [], [], [], []
    conf_mat = []
    for name, model in models.items():
        print(f"\nFitting {name}...")

        model.fit(X, y)
        y_pred = model.predict(X_v)

        acc = accuracy_score(y_v, y_pred)
        accuracy.append(acc)
        precision = precision_score(y_v, y_pred)
        prec.append(precision)
        recall = recall_score(y_v, y_pred)
        rec.append(recall)
        auc_ = roc_auc_score(y_v, y_pred)
        auc.append(auc_)
        f1_ = f1_score(y_v, y_pred)
        f1.append(f1_)
        conf_mat.append(confusion_matrix(y_v, y_pred))

        performance_temp = [acc, precision, recall, auc_, f1_]
        print(f"Performance: acc - precision, recall, auc, f1\n {performance_temp}")
        print(f"Confusion matrix: {confusion_matrix(y_v, y_pred)}")
    
    perf_df = pd.DataFrame()
    perf_df["model"] = list(models.keys())
    perf_df["accuracy"] = accuracy
    perf_df["precision"] = prec
    perf_df["recall"] = rec
    perf_df["f1"] = f1
    perf_df["auc"] = auc
    perf_df["confusion_matrix"] = conf_mat

    perf_df.to_csv("other_models_perf.csv")

if __name__ == "__main__":
    main()
