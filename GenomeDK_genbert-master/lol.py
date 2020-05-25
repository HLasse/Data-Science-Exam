import pandas as pd
import os

# set WD
# home = Path.home()
# wd = os.path.abspath(os.path.join(home, 'NLPPred'))
# os.chdir(wd)

def check_shape(ends = ".fam"):
    files = [f for f in os.listdir()
               if f.endswith(ends)]
    for f in files:
        print(f"shape of {f}")
        print(pd.read_csv(f, header=None, sep=" ").shape)

def get_n_participants():
    files = [f for f in os.listdir()
               if f.endswith(".fam")]

    total = 0
    for f in files:
        print(f"n subjects in {f}")
        n_sub = pd.read_csv(f, header=None, sep=" ").shape[0]
        print("  ",  n_sub)
        total += n_sub

    print("total: ", total)

def get_participant_split():
    files = [f for f in os.listdir()
               if f.endswith(".fam")]

    total = 0
    total_1 = 0
    total_2 = 0
    for f in files:
        print(f"n subjects in {f}")
        counts = pd.read_csv(f, header=None, sep=" ")[5].value_counts()
        total += sum(counts)
        total_1 += counts[1]
        total_2 += counts[2]

        print("  1: ", counts[1])
        print("  2: ", counts[2])

    print("total: ", total)
    print("total_1: ", total_1)
    print("total_2: ", total_2)



os.chdir("/faststorage/project/NLPPred/inter_data")
os.chdir("/faststorage/project/NLPPred")


get_participant_split()

get_n_participants()

check_shape()

