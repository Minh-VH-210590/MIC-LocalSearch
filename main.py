import csv
import time
import sys
import os

import numpy as np
import pandas as pd

from binlib import mic

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
# print(df.head())

start_time = time.time()
S0 = mic.genSolution(df, X, Y)
S = mic.MIC_LocalSearch(df, X, Y, T, S0)
print("MIC=", S.score)
print("Runtime:", time.time() - start_time)

extras = [f"Runtime: {time.time() - start_time}"]

S._export(LOG, extras)
with open(CSV, 'a') as file:
    file.write(f'{DATASET},{X},{Y},mic-ls,{S.score},{T}\n')
    file.close()