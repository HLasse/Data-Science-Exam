"""
quit()
srun --mem=64g --pty /bin/bash
conda activate tf-gpu15
python
"""

## Models not using the encoded representation
import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import tensorflow as tf

def setwd(path="GenomeDK_genbert"):
    home = Path.home()
    wd = os.path.abspath(os.path.join(home, 'NLPPred', path))
    os.chdir(wd)

def rem_layers(n, best_model):
    model = tf.keras.Sequential()
    for layer in best_model.layers[:n+1]:
        # print(layer)
        model.add(layer)
    return model

def main():
    setwd()

    # load model
    best_model = tf.keras.models.load_model('wandb/old/dryrun-20200521_022220-3vkn4r74/model-best.h5')
    model = rem_layers(n=11, best_model=best_model)

    # load data
    setwd("inter_data")
    with open("x_test.npy", 'rb') as f:
        X = np.load(f)
        print(X.shape)
    X.shape = (*X.shape, 1)

    with open("x_val.npy", 'rb') as f:
        X_val = np.load(f)
        print(X_val.shape)
    X_val.shape = (*X_val.shape, 1)

    # Apply encoder
    reduced_X = model.predict(X)
    np.save("encoded_test_data.npy", reduced_X)

    reduced_X_val = model.predict(X_val)
    np.save("encoded_val_data.npy", reduced_X_val)


if __name__ == "__main__":
    main()