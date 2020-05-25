"""
quit()
srun --mem=64g --pty /bin/bash
conda activate tf-gpu15
python
"""
import os
from pathlib import Path

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def setwd():
    home = Path.home()
    wd = os.path.abspath(os.path.join(home, 'NLPPred', "inter_data"))
    os.chdir(wd)


def load_fam(f):
    key = f.split(".")[0].split("_")[1]
    y = pd.read_csv("../mhc" + key+".fam", header=None, sep=" ")
    y = y[y.columns[5]].to_numpy()
    return y


def load_data(control_from_test=True):
    files = [f for f in os.listdir()
             if f.endswith(".sp") and f.split(".")[0][-3:] != "abd"]

    added_control = control_from_test
    # for low let's just use one file
    for f in files:
        df = pd.read_csv(f, header=None, sep=" ")
        x = df.to_numpy()
        y = load_fam(f)
        if y.shape[0] != x.T.shape[0]:
            print(f".fam have shape {y.shape} and .sp have shape {x.shape} \
                adjusting size to match")
            x = x[:, :y.shape[0]]
        control = x.T[y == 1, :]
        affected = x.T[y == 2, :]

        if control.shape[0] == 7508 and (added_control is False):
            # take subset of control from test set
            indices = np.arange(len(y[y == 1]))
            rng = np.random.RandomState(1994)  # old 42
            rng.shuffle(indices)

            add = np.concatenate((control[indices[1963:], :], affected),
                                 axis=0)
            add_y = np.concatenate((y[y == 1][indices[1963:]], y[y == 2]),
                                   axis=0)

            y_test = y[y == 1][indices[:1963]]
            x_test = control[indices[:1963], :]
        else:
            add = affected
            add_y = y[y == 2]

        if f == files[0]:
            res_x = add
            res_y = add_y
        else:
            res_x = np.concatenate((res_x, add), axis=0)
            res_y = np.concatenate((res_y, add_y), axis=0)

    if control_from_test is True:
        x_test, y_test = load_test_data(return_control=True)
    else:
        x_t, y_t = load_test_data(return_control=False)
        x_test = np.concatenate((x_t, x_test), axis=0)
        y_test = np.concatenate((y_t, y_test), axis=0)

    return res_x, res_y, x_test, y_test


def load_test_data(return_control=True):
    files = [f for f in os.listdir()
             if f.endswith(".sp") and f.split(".")[0][-3:] == "abd"]
    f = files[0]
    df = pd.read_csv(f, header=None, sep=" ")
    x = df.to_numpy()
    y = load_fam(f)
    if y.shape[0] != x.T.shape[0]:
        print(f".fam have shape {y.shape} and .sp have shape {x.shape} \
            adjusting size to match")
        x = x[:, :y.shape[0]]
    if return_control:
        return x.T, y
    return x.T[y == 2, :], y[y == 2]


def one_hot_encode(df):
    # Converting NAs to -1
    df = np.nan_to_num(df, nan=-1)
    enc = OneHotEncoder()
    return enc.fit_transform(df)


def encode_snp(a, na_to=[1, 0]):
    """
    encode SNP using coding scheme
    NA = 1, 0
    0 = 0, 0
    1 = 0, 1
    2 = 1, 1

    Example
    >>> a = np.random.randint(0,3,(3, 10))
    >>> a=a.astype(float)
    >>> a[1,1] = np.nan
    >>> # encode_snp(a)
    """
    out = np.zeros(a.shape + (2,), dtype=int)

    d = {0: [0, 0], 1: [0, 1],
         2: [1, 1], None: na_to}
    for i in [0, 1, 2]:
        cond = a == i
        n = a[cond].shape[0]
        x = np.repeat(d[i], n)
        x.shape = (2, n)
        out[cond, :] = x.T
    cond = np.isnan(a)
    n = a[cond].shape[0]
    x = np.repeat(d[None], n)
    x.shape = (2, n)
    out[cond, :] = x.T
    return out


if __name__ == "__main__":
    setwd()

    x_train, y_train, x_test, y_test = load_data()
    x_test_, x_val, y_test_, y_val = \
        train_test_split(x_test, y_test,
                         test_size=0.20, random_state=1994)  # old 42

    # balance dataset
    n_diagnosed = len(y_test_[y_test_ == 2])
    y_test1 = y_test_[y_test_ == 1][:n_diagnosed]
    x_test1 = x_test_[y_test_ == 1, :][:n_diagnosed, :]
    x_test_ = np.concatenate([x_test1, x_test_[y_test_ == 2, :]])
    y_test_ = np.concatenate([y_test1, y_test_[y_test_ == 2]])

    n_diagnosed = len(y_val[y_val == 2])
    y_val1 = y_val[y_val == 1][:n_diagnosed]
    x_val1 = x_val[y_val == 1, :][:n_diagnosed, :]
    x_val = np.concatenate([x_val1, x_val[y_val == 2, :]])
    y_val = np.concatenate([y_val1, y_val[y_val == 2]])
    # normalize to 0-1
    y_val = y_val - 1
    y_test_ = y_test_ - 1
    y_train = y_train - 1

    # print(len(y_val[y_val == 1]), len(y_val[y_val == 0]), " - diff:", len(y_val)-len(y_val[y_val == 1])-len(y_val[y_val == 0]))
    # print(len(y_test_[y_test_ == 1]), len(y_test_[y_test_ == 0]), " - diff:", len(y_test_)-len(y_test_[y_test_ == 1])-len(y_test_[y_test_ == 0]))

    # fix size to match network
    x_train = x_train[:, :18873]
    x_test_ = x_test_[:, :18873]
    x_val = x_val[:, :18873]

    # One hot encoding
    enc = OneHotEncoder(handle_unknown='ignore')
    x_train_1h = np.nan_to_num(x_train, nan=-1)
    x_test_1h = np.nan_to_num(x_test_, nan=-1)
    x_val_1h = np.nan_to_num(x_val, nan=-1)
    enc.fit(x_train_1h)
    x_train_1h = enc.transform(x_train_1h)
    x_test_1h = enc.transform(x_test_1h)
    x_val_1h = enc.transform(x_val_1h)

    # name into correct encoding
    x_train = encode_snp(x_train)
    x_test_ = encode_snp(x_test_)
    x_val = encode_snp(x_val)

    # print(x_test.shape, y_test.shape)
    np.save("x_test.npy", x_test_)
    np.save("y_test.npy", y_test_)
    np.save("x_train.npy", x_train)
    np.save("y_train.npy", y_train)
    np.save("x_val.npy", x_val)
    np.save("y_val.npy", y_val)

    pickle.dump(x_train_1h, open("x_train_1h.dat", 'wb'), protocol=4)
    pickle.dump(x_test_1h, open("x_test_1h.dat", 'wb'), protocol=4)
    pickle.dump(x_val_1h, open("x_val_1h.dat", 'wb'), protocol=4)

    # with open("x_test.npy", 'rb') as f:
    #     a = np.load(f)
