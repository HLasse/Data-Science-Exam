"""
quit()
srun --mem=64g --pty /bin/bash
conda activate tf-gpu15
python

# load model
new_model = tf.keras.models.load_model('saved_model/my_model')
# Check its architecture
new_model.summary()

# add new top - you need ot fix how much to remove
model = Sequential()
for layer in new_model.layers[:-1]: # go through until last layer
    model.add(layer)
model.add(Dense(3, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy')
"""
import os
from pathlib import Path
import logging
import argparse

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

import wandb
from wandb.keras import WandbCallback

# update config using flags
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=10)
parser.add_argument('-l', '--lr', type=float, default=0.001)
parser.add_argument('-dr', '--dropout', type=float, default=0.0)
parser.add_argument('-dl', '--dense_layers', type=float, default=2)
parser.add_argument('-df', '--dense_factor', type=float, default=2)
parser.add_argument('-t', '--transfer', type=str, default="False")
parser.add_argument('-r', '--regularizer', type=str, default="False")
args = vars(parser.parse_args())

for flag in ["transfer", "regularizer"]:
    if flag in args:
        if args[flag].lower() == "true":
            args[flag] = True
        elif args[flag].lower() == "false":
            args[flag] = False
        else:
            raise ValueError(f"invalid {flag} flag")

# Set hyperparameters, which can be overwritten with a W&B Sweep
hyperparameter_defaults = dict(
    dropout=args['dropout'],
    reg=args['regularizer'],
    learn_rate=args['lr'],
    epochs=args['epochs'],
    dlayers=args['dense_layers'],
    df=args['dense_factor'],
    transfer=args['transfer'],
)

# Initialize wandb
os.environ['WANDB_MODE'] = 'dryrun'
name = "transfer learning"
project = "TransferSNP"
wandb.init(project=project, name=name,
           config=hyperparameter_defaults)
config = wandb.config

print(hyperparameter_defaults)


def setwd(path="GenomeDK_genbert"):
    home = Path.home()
    wd = os.path.abspath(os.path.join(home, 'NLPPred', path))
    os.chdir(wd)


def add_pred_layers(model):
    if config.reg:
        reg = "l1"
    else:
        reg = None
    if config.dropout:
        model.add(Dropout(config.dropout, name="dropout_c"))
    if config.dlayers > 2:
        model.add(Dense(128*config.df, activation='relu',
                        kernel_regularizer=reg, name="PredDense3"))
    model.add(Dense(64*config.df, activation='relu',
                    kernel_regularizer=reg, name="PredDense2"))
    model.add(Dense(32*config.df, activation='relu',
                    kernel_regularizer=reg,
                    name="PredDense1"))
    model.add(Dense(1, activation='sigmoid', name="prediction"))
    return model


def rem_layers(n, best_model):
    model = tf.keras.Sequential()
    for layer in best_model.layers[:n+1]:
        # print(layer)
        model.add(layer)
    return model


def train(X, y, model, validation):
    adam = tf.keras.optimizers.Adam(learning_rate=config.learn_rate,
                                    amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=adam,
                  metrics=['accuracy', 'AUC', 'Precision', 'Recall'])
    hist = model.fit(X, y, validation_data=validation, epochs=config.epochs,
                     callbacks=[WandbCallback()])
    # model.save(os.path.join(wandb.run.dir, config.model + "_model.h5"))
    return hist.history


def send_mail(subject, message,
              reciever=["anon@gmail.com",
                        "anon@gmail.com"]):
    for r in reciever:
        cmd = 'mail -s "' + subject + '" "' + r + '" <<< "' + message + '"'
        os.system(cmd)


def reset_weights(model):
    print("reset_weights called")
    session = tf.compat.v1.keras.backend.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session)
    return model


def main():
    start_rep = f"{project}: process {name}: have started"
    # send_mail(start_rep, start_rep)
    setwd()

    # load model
    best_model = tf.keras.models.load_model('wandb/old/dryrun-20200521_022220-3vkn4r74/model-best.h5')
    # best_model.summary()
    model = rem_layers(n=11, best_model=best_model)
    model = add_pred_layers(model)
    # print(model.summary())

    # load data
    setwd("inter_data")
    with open("x_test.npy", 'rb') as f:
        X = np.load(f)
        print(X.shape)
    with open("y_test.npy", 'rb') as f:
        y = np.load(f)
        print(y.shape)
    X.shape = (*X.shape, 1)
    with open("x_val.npy", 'rb') as f:
        X_v = np.load(f)
        print(X_v.shape)
    with open("y_val.npy", 'rb') as f:
        y_v = np.load(f)
        print(y_v.shape)
    X_v.shape = (*X_v.shape, 1)

    if config.transfer is False:
        model = reset_weights(model)

    try:
        perf = train(X, y, model, validation=(X_v, y_v))
        report = f"{project}: process {name}: done"
        report_long = report + f"\n\n The best {str(perf)}"
    except Exception as e:
        report = f"{project}: process {name}: failed"
        report_long = report + f" with error {e}\n " + \
            f"Logging:\n{logging.exception('message')}"
    # send_mail(report, report_long)


if __name__ == "__main__":
    main()
