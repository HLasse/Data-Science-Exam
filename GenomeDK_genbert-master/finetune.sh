#!/bin/bash
#SBATCH --partition gpu
#SBATCH --mem 64g
#SBATCH -c 8
#SBATCH --gres=gpu:1

python finetune.py -t TRUE