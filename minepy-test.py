import csv
import time
import sys
import os

import numpy as np
import pandas as pd

from minepy import MINE

DATA = sys.argv[1]
DATASET = sys.argv[2]
FEATURE_X = sys.argv[3]
FEATURE_Y = sys.argv[4]
T = int(sys.argv[5])

LOG = os.path.join('logs', f'{DATASET}-{FEATURE_X}-{FEATURE_Y}.txt')
CSV = 'report.csv'

df = pd.read_csv(os.path.join(DATA, DATASET))
df.replace(r'\W', np.nan, regex=True, inplace= True)
X = FEATURE_X
Y = FEATURE_Y
df = df.dropna(subset= [X], inplace= False, ignore_index= True)
df = df.dropna(subset= [Y], inplace= False, ignore_index= True)

mine_approx = MINE(alpha=0.6, c=15, est="mic_approx")
mine_approx.compute_score(df[X], df[Y])
with open(CSV, 'a') as file:
    file.write(f'{DATASET},{X},{Y},mic-approx,{mine_approx.mic()},{T}\n')
    file.close()

mine_e = MINE(alpha=0.6, c=15, est="mic_e")
mine_e.compute_score(df[X], df[Y])
with open(CSV, 'a') as file:
    file.write(f'{DATASET},{X},{Y},mic-e,{mine_approx.mic()},{T}\n')
    file.close()