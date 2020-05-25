"""
quit()
srun --mem=64g --pty /bin/bash
conda activate umap
python
"""


## Models not using the encoded representation
import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import umap

def setwd(path="inter_data"):
    home = Path.home()
    wd = os.path.abspath(os.path.join(home, 'NLPPred', path))
    os.chdir(wd)

def load_data():
    with open("encoded_test_data.npy", "rb") as f:
        train = np.load(f)
    with open("encoded_val_data.npy", "rb") as f:
        test = np.load(f)
    return (train, test)


def make_umap_emb(x):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(x)
    return embedding

def make_vis(embedding, train = True):
    if train:
        with open("y_test.npy", 'rb') as f:
            y = np.load(f)
            lab = "training"
    else: 
        with open("y_val.npy", "rb") as f:
            y = np.load(f)
            lab = "test"

    plt.scatter(embedding[:, 0], embedding[:, 1], s= 10, c=[sns.color_palette()[x] for x in y])
    plt.gca().set_aspect('equal', 'datalim')
    plt.savefig("umap_proj_" + lab + ".png")


def main():
    setwd()
    train, test = load_data()
    emb_train = make_umap_emb(train)
    emb_test = make_umap_emb(test)
    make_vis(emb_train)
    make_vis(emb_test, train = False)

if __name__ == "__main__":
    main()