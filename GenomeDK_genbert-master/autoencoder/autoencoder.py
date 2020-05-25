"""

srun --mem=64g --gres=gpu:1 -p gpu --pty /bin/bash
conda activate tf-gpu15
python


python autoencoder.py -f 1 -cl 2 -d True

srun --mem=64g --pty /bin/bash
conda activate tf-gpu15
python
"""
import os
import argparse
from pathlib import Path
import logging
# import time

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Reshape, \
     MaxPool2D, UpSampling2D, Dense, Flatten, Dropout

import wandb
from wandb.keras import WandbCallback


if int(tf.__version__.split(".")[0]) < 2:
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
else:
    print(tf.config.list_physical_devices('GPU'))

# update config using flags
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=10)
parser.add_argument('-l', '--lr', type=float, default=0.001)
parser.add_argument('-f', '--filter_factor', type=float, default=1)
parser.add_argument('-cl', '--conv_layers', type=float, default=2)
parser.add_argument('-d', '--dense', type=str, default="False")
parser.add_argument('-de', '--denoise', type=float, default=0)
args = vars(parser.parse_args())

if "dense" in args:
    if args['dense'].lower() == "true":
        args['dense'] = True
    elif args['dense'].lower() == "false":
        args['dense'] = False
    else:
        raise ValueError("invalid dense flag")


# Set hyperparameters, which can be overwritten with a W&B Sweep
hyperparameter_defaults = dict(
    learn_rate=args['lr'],
    epochs=args['epochs'],
    layers=args['conv_layers'],
    ff=args['filter_factor'],
    dense=args['dense'],
    denoise=args['denoise']
)

# Initialize wandb
os.environ['WANDB_MODE'] = 'dryrun'
name = f"m_D{str(args['dense'])}_C{str(args['conv_layers'])}_\
    F{str(args['filter_factor'])}_DN{str(args['denoise'])}"
project = "ConvSNP"
wandb.init(project=project, name=name,
           config=hyperparameter_defaults)

config = wandb.config


def setwd():
    home = Path.home()
    wd = os.path.abspath(os.path.join(home, 'NLPPred', "inter_data"))
    os.chdir(wd)


def send_mail(subject, message,
              reciever=["anon@gmail.com",
                        "anon@gmail.com"]):
    for r in reciever:
        cmd = 'mail -s "' + subject + '" "' + r + '" <<< "' + message + '"'
        os.system(cmd)


def train(X, model):
    adam = tf.keras.optimizers.Adam(learning_rate=config.learn_rate,
                                    amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=adam,
                  metrics=['accuracy'])
    hist = model.fit(X, X, validation_split=0.2, epochs=config.epochs,
                     callbacks=[WandbCallback()])
    model.save(os.path.join(wandb.run.dir, f"{name}_model.h5"))
    return hist.history


def model_CAE(X):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=X.shape[1:]))
    # Denoising
    if config.denoise:
        model.add(Dropout(config.denoise))

    # Encoding
    model.add(Conv2D(filters=int(75*config.ff), kernel_size=(60, 2),
                     activation='relu',
                     padding='same'))
    model.add(MaxPool2D(pool_size=(3, 2)))
    if config.layers == 4:
        model.add(Conv2D(filters=int(50*config.ff), kernel_size=(9, 1),
                         activation='relu',
                         padding='same'))
        model.add(MaxPool2D(pool_size=(3, 1)))
    if config.layers > 2:
        model.add(Conv2D(filters=int(20*config.ff), kernel_size=(9, 1),
                         activation='relu',
                         padding='same'))
        model.add(MaxPool2D(pool_size=(3, 1)))
    model.add(Conv2D(filters=1, kernel_size=(9, 1),
                     activation='relu',
                     padding='same'))
    model.add(MaxPool2D(pool_size=(3, 1)))
    model.add(Conv2D(filters=1, kernel_size=(9, 1),
                     activation='relu',
                     padding='same'))

    if config.dense:
        out_shape = int(18873/(3**config.layers))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Dense(out_shape))
        model.add(Reshape((out_shape, 1, 1)))

    # Decoding
    model.add(UpSampling2D(size=(3, 1)))
    if config.layers > 2:
        model.add(Conv2D(filters=int(20*config.ff), kernel_size=(9, 1),
                         activation='relu',
                         padding='same'))
        model.add(UpSampling2D(size=(3, 1)))
    if config.layers == 4:
        model.add(Conv2D(filters=int(50*config.ff), kernel_size=(9, 1),
                         activation='relu',
                         padding='same'))
        model.add(UpSampling2D(size=(3, 1)))
    model.add(Conv2D(filters=int(75*config.ff), kernel_size=(9, 1),
                     activation='relu',
                     padding='same'))
    model.add(UpSampling2D(size=(3, 2)))
    model.add(Conv2D(filters=1, kernel_size=(100, 2),
                     activation='sigmoid',
                     padding='same'))
    return model


def main():
    setwd()
    with open("x_train.npy", 'rb') as f:
        X = np.load(f)
    X.shape = (*X.shape, 1)

    try:
        model = model_CAE(X)
        print(model.summary())

        start_rep = f"{project}: process {name}: have started"
        send_mail(start_rep, start_rep)
        perf = train(X, model)
        report = f"{project}: process {name}: done"
        report_long = report + f"\n\n The best {str(perf)}"
    except Exception as e:
        report = f"{project}: process {name}: failed"
        report_long = report + f" with error {e}\n " + \
            f"Logging:\n{logging.exception('message')}"
    send_mail(report, report_long)


if __name__ == "__main__":
    main()
