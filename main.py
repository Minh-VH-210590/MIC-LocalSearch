import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

from binlib import mic

# Generate sample, following: https://minepy.readthedocs.io/en/latest/python.html#first-example
# x = np.linspace(0, 1, 1000)
# y = np.sin(10 * np.pi * x) + x

# np.random.seed(0)
# dy = y + np.random.uniform(-1, 1, x.shape[0]) # add some 

# D = np.stack((x, y, dy), axis = 1)
# df = pd.DataFrame(D, header= False)
# df.to_csv('sample.csv')

df = pd.read_csv('sample.csv')
df = df[0:150]
X = 'x'
Y = 'y'

S0 = mic.genSolution(df, X, Y)
S = mic.MIC_LocalSearch(df, X, Y, 30, S0)
print("Without noise, MIC=", S.score)

Y = 'dy'
S0 = mic.genSolution(df, X, Y)
S = mic.MIC_LocalSearch(df, X, Y, 30, S0)
print("With noise, MIC=", S.score)