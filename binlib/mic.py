import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

from binlib import utils, metrics, binalgo, localSearch

def genSolution(df, X, Y, nX= 2, Bn= None, exact= False, fixcol= None):
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
    archive = []

    # Exhaustive search
    if exact:
        assert fixcol == 0 or fixcol == 1
        startTime = time.time()
        scoreList = [-1]
        bestSol = None
        mxScore = 0
        for i in range(len(maskX)):
            if i % 10 == 0:
                print(f"Generating {i}-th solution with fixed column {X}. Current record: {mxScore}")
            new_mask = np.zeros_like(valX).astype(int)
            new_mask[i] = 1
            score, splt = Opt_YbyX(df, X, Y, splitX = utils.mask2Split(new_mask, valX), Bn= Bn)
            maskY = utils.split2Mask(splt, valY)

            if fixcol == 0:
                nS = localSearch.Solution(maskX, maskY, fixCol = fixcol, score = score)
            else:
                nS = localSearch.Solution(maskY, maskX, fixCol = fixcol, score = score)
            archive.append(nS._encode())

            if score > mxScore:
                bestSol = nS._copy()
                mxScore = score

            scoreList.append(score)
        print(f'Best score: {max(scoreList)}, Runtime: {time.time() - startTime}')
        return scoreList, bestSol, set(archive)

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

    # print(f'{X}={nX}; {Y}={nY}')

    if nX <= 1 or nY <= 1:
        return 0, []

    oldX = df[X].copy()
    df[X] = dX_val # Replace df[X] by its discretized version

    # Conisder discretized X as the target class for Y
    val, freq, _ = utils.makePrebins(df, Y, X, num_classes= nX)
    score, splitY, opt_nY = binalgo.scoreDP(val, freq, R = nY, mic = True, nX = nX)
    score = score

    df[X] = oldX.copy() # Restore original df[X]
    return score, splitY

def MIC_LocalSearch(df, X, Y, T, S:localSearch.Solution, nseed= 30, Bn= None, init_archive= None):
    '''
    Calculate MIC using Local Search.

    Input:
    -----
    df  {pd.DataFrame}  : Dataset object
    X {String}      : Feature #1 name
    Y {String}      : Feature #2 name
    T {int}         : Number of iteration
    S {localSearch.Solution} : Initial solution

    Ouput:
    -----
    S {localSearch.Solution} : Final solution
    '''
    start_time = time.time()
    # Preparation
    valX = utils.makeVal(df, X)
    valY = utils.makeVal(df, Y)
    if init_archive is None:
        archive = set()
    else:
        archive = set(init_archive.copy())
    archive.add(S._encode())

    if Bn is None:
        Bn = int(len(df) ** 0.6)

    bestS = localSearch.Solution(S.maskX, S.maskY, S.fixCol, S.score)
    current_type = 'None'

    tabu_flag = False

    # Iterations
    for t in range(T):
        # Choose a new seed every 10 iterations
        if t % 10 == 0 and t > 0:
            S = genSolution(df, X, Y, nX= int(t/5) + 1)

        # Set fixed column
        fixed_col = X
        if S.fixCol == 1:
            fixed_col = Y
        print(f'Iteration {t}: Score = {S.score}, fixed column = {fixed_col}, operation = {current_type}')
        
        # Generate switch solution
        switchS = localSearch.Solution(S.maskX, S.maskY, 1 - S.fixCol, S.score)

        # Generate neighbors from root solutions (original and switched solutions)
        nS_shortlist = localSearch.exhaustMutate(S, df, X, Y, Bn) + localSearch.exhaustMutate(switchS, df, X, Y, Bn)
        nS_list = []

        # Remove visited candidates
        for (nS, operation_type) in nS_shortlist:
            if nS._encode() in archive:
                continue
            nS_list.append((nS._copy(), operation_type))

        # Flag for tabu search
        flag = False
        idx = np.arange(len(nS_list))
        print(len(nS_list))
        np.random.shuffle(idx)

        if nseed > 0:
            idx_list = idx[0:min(nseed, len(idx))]
        else:
            # If nseed < 0, traverse through ALL shortlisted neighbours
            idx_list = idx[0:len(idx)]

        candidates = [nS_list[x] for x in idx_list]
        candidates.append((localSearch.Switch(S, df, X, Y), 'Switch'))

        for (nS, operation_type) in candidates:
            if nS._encode() in archive:
                continue
            archive.add(nS._encode())

            # Update one feature w.r.t the fixed feature
            if operation_type != 'Switch':
                if nS.fixCol == 0:
                    # Convert mask to split scheme
                    spltX = utils.mask2Split(nS.maskX, valX)
                    # Optimize axis
                    score, splt = Opt_YbyX(df, X, Y, spltX)
                    nS.score = score
                    # Obtain mask after optimization
                    nS.maskY = utils.split2Mask(splt, valY)
                else:
                    spltY = utils.mask2Split(nS.maskY, valY)
                    score, splt = Opt_YbyX(df, Y, X, spltY)
                    nS.score = score
                    nS.maskX = utils.split2Mask(splt, valX)

            # Update solution
            if nS.score > S.score:
                S = nS._copy()
                flag = True
                tabu_flag = False
                current_type = operation_type

            # Update current solutions
            if nS.score > bestS.score:
                bestS = nS._copy()
                flag = True
                tabu_flag = False
                current_type = operation_type
            
            # Tabu search
            elif tabu_flag:
                # Metropolis search
                coin = np.random.binomial(n=1, p=np.exp(nS.score - S.score)/(T+1))
                if coin == 1:
                    S = nS._copy()
                    flag = True
                    current_type = operation_type
                
        if not flag:
            print('Activate Tabu search.')
            tabu_flag = True
            # print(f'Final solution: Score = {S.score}, fixed column = {fixed_col}, operation = {current_type}')
            # break
        else:
            print('Deactivate Tabu search')
            tabu_flag = False


    print(f'Final score: {bestS.score}, Runtime: {time.time() - start_time}')
    return bestS