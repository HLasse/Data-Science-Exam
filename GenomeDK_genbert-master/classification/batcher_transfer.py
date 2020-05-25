"""
script for batching using slurm

Example:
python finetune.py -t False
"""
import os

init_batch = """#!/bin/bash
#SBATCH --partition gpu
#SBATCH --mem 64g
#SBATCH -c 8
#SBATCH --gres=gpu:1"""

init_batch = """#!/bin/bash
#SBATCH --mem 64g
#SBATCH -c 16"""

grid = []
for dl in [2, 3]:
    for t in [False, True]:
        for df in [1, 2]:
            for dr in [0, 0.1, 0.5]:
                for reg in [True, False]:
                    for lr in [0.001, 0.0001]:
                        grid.append((dl, t, df, dr, reg, lr))


# learning rate, regularization (L1), Dropout

# make .sh scripts
for g in grid:
    dl, t, df, dr, reg, lr = g
    python_cmd = f"python finetune.py -t {t} -dl {dl} -df {df} -l {lr} -dr {dr} -r {reg}"

    filename = f"run_finetune_T{t}_DL{dl}_DF{df}_{lr}_DR{dr}_R{reg}.sh"
    with open(filename, "w") as f:
        f.write(init_batch + "\n" + python_cmd)
    os.system(f"sbatch {filename} -A NLPPred")
