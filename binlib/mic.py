import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from binlib import utils, metrics, binalgo, localSearch

def genSolution(df, X, Y, nX= 2, Bn= None):
    '''
    Generate a random solution with X being fixed and Y is optimized w.r.t X

    Input:
    -----
    df {pd.DataFrame}   : Dataset
    X {String}  : Fixed feature
    Y {String}  : Optimized feature
    nX {int}    : The number of prebin chosen (randomly) over X
    Bn {int}    : The grid budget nX * nY

    Output:
    -----
    S {localSearch.Solution}    : The generated solution
    S.fixCol = 0 as X is the fixed axis
    '''
    if Bn is None:
        Bn = len(df) ** 0.6

    valX = utils.makeVal(df, X)
    valY = utils.makeVal(df, Y)
    maskX = np.zeros_like(valX).astype(int)

    # Randomly bin X
    for i in range(nX-1):
        localSearch.turnOn(maskX)
    maskX[-1] = 1

    # Optimize Y-axis
    score, splt = Opt_YbyX(df, X, Y, splitX = utils.mask2Split(maskX, valX), Bn= Bn)
    maskY = utils.split2Mask(splt, valY)

    S = localSearch.Solution(maskX, maskY, 0, score)
    return S


def Opt_YbyX(df, X, Y, splitX, Bn= None):
    '''
    Return candidate MIC-score given a fixed partition over X and a grid budget: xy <= Bn
    '''
    # Initialize Bn
    if Bn is None:
        Bn = len(df) ** 0.6 # Default value of Bn = n^0.6
    
    dX_val = utils.discretizeFea(df, X, splitX)
    nX = max(dX_val) + 1

    if nX == 0:
        return 0, []

    nY = int(Bn / nX)

    print(f'{X}={nX}; {Y}={nY}')

    if nX <= 1 or nY <= 1:
        return 0, []

    oldX = df[X]
    df[X] = dX_val # Replace df[X] by its discretized version

    # Conisder discretized X as the target class for Y
    val, freq, _ = utils.makePrebins(df, Y, X, num_classes= nX)
    score, splitY, opt_nY = binalgo.scoreDP(val, freq, R = nY)
    score = score / np.log(min(nX, opt_nY))

    df[X] = oldX # Restore original df[X]
    return score, splitY

def MIC_LocalSearch(df, X, Y, T, S:localSearch.Solution, p= [0.5, 0.5, 0.5, 0.9], maxD= 3):
    '''
    Calculate MIC using Local Search.

    Input:
    -----
    df  {pd.DataFrame}  : Dataset object
    X {String}      : Feature #1 name
    Y {String}      : Feature #2 name
    T {int}         : Number of iteration
    S {localSearch.Solution} : Initial solution
    p {array of length 4}    : Mutation configs
    maxD {int}               : Maximal split point moving distance

    Ouput:
    -----
    S {localSearch.Solution} : Final solution
    '''
    # Preparation
    valX = utils.makeVal(df, X)
    valY = utils.makeVal(df, Y)
    archive = set()
    archive.add(S._encode())

    # Iterations
    for t in range(T):
        fixed_col = X
        if S.fixCol == 1:
            fixed_col = Y
        print(f'Iteration {t}: Score = {S.score}, fixed column = {fixed_col}')

        # Mutation
        nS = localSearch.Mutate(S, p, maxD)
        patience = 10
        while nS._encode() in archive:
            nS = localSearch.Mutate(S, p, maxD)
            patience -= 1
            if patience == 0:
                break
        if patience == 0:
            continue
        archive.add(nS._encode())

        # Update one feature w.r.t the fixed feature
        if nS.fixCol == 0:
            spltX = utils.mask2Split(nS.maskX, valX)
            score, splt = Opt_YbyX(df, X, Y, spltX)
            nS.score = score
            nS.maskY = utils.split2Mask(splt, valY)
        else:
            spltY = utils.mask2Split(nS.maskY, valY)
            score, splt = Opt_YbyX(df, Y, X, spltY)
            nS.score = score
            nS.maskX = utils.split2Mask(splt, valX)

        # Update solution
        if nS.score > S.score:
            S.maskX = nS.maskX
            S.maskY = nS.maskY
            S.score = nS.score
            S.fixCol = nS.fixCol

    print(f'Final score: {S.score}')
    return S