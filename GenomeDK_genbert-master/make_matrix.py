"""
Wrapper script for converting .bim (binary) to .sp (matrix)

Author: K. Enevoldsen
"""

from pathlib import Path
import os


# set WD
home = Path.home()
wd = os.path.abspath(os.path.join(home, 'NLPPred'))
os.chdir(wd)

files = [f for f in os.listdir() if f.endswith(".bim")]
new_files = [f.split(".")[0][3:] for f in os.listdir()
             if f.startswith("new") and f.endswith(".sp")]
files = [f for f in files if f.split(".")[0][3:] not in new_files]


#awk '(NR==FNR){arr[$2];next}($2 in arr){print $2}' > over.all mhcabd.bim mhcabe.bim mhcabc.bim mhcabg.bim mhcabi.bim mhcuvps.bim mhcuvis.bim mhcuvk.bim mhcubvbuc.bim mhcuvms.bim mhcuvbo.bim mhccel.bim

for f in files:
    f = f.split(".")[0]
    new_f = "intersect_" + f[3:]
    cmd = "./ldak5.linux --make-sp " + new_f + " --bfile " + f + " --extract over.all"
    os.system(cmd)
    print(f"{f} done")


"./ldak5.linux --make-sp " + "inter_test_abc" + " --bfile " + "mhcabc" + " --extract over.abc.abd"
"./ldak5.linux --make-sp " + "inter_test_abd" + " --bfile " + "mhcabd" + " --extract over.abc.abd"

os.system(test)

#
#mv dir/new* /path/to/destination

