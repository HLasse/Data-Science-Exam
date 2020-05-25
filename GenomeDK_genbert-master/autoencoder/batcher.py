"""
script for batching using slurm

Example:
python autoencoder.py -f 2 -cl 4 -d True
"""
import os

init_batch = """#!/bin/bash
#SBATCH --partition gpu
#SBATCH --mem 64g
#SBATCH -c 8
#SBATCH --gres=gpu:1"""

grid = []
for l in range(2, 5):
    for d in [False]:
        for f in [1, 2]:
            for de in [0.1]:
                grid.append((l, d, f, de))


# make .sh scripts
for g in grid:
    c, d, ff, de = g
    if d is True:
        python_cmd = f"python autoencoder.py -f {ff} -cl {c} -d True -de {de}"
    elif d is False:
        python_cmd = f"python autoencoder.py -f {ff} -cl {c} -de {de}"

    filename = f"run_autoencoder_D{str(d)}_C{str(c)}_F{str(ff)}_DN{str(de)}.sh"
    with open(filename, "w") as f:
        f.write(init_batch + "\n" + python_cmd)
    os.system(f"sbatch {filename} -A NLPPred")
