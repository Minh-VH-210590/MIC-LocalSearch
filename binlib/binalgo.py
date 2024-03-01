import numpy as np
import pandas as pd

from binlib import utils, metrics

def Opt(val, cand, mode):
    '''
    Update 'val' using 'cand', regarding 'mode'

    Output:
    -----
    winner : 'val' or 'cand', depending on 'mode'
    update {boolean} : whether to update val by cand
    '''
    if mode == 'min':
        if val > cand:
            return cand, True
        return val, False
    
    if mode == 'max':
        if val < cand:
            return cand, True
        return val, False

def scoreDP(val, freq, mode = 'max', metric = 'mi', L = 2, R = 15):
    '''
    Proposed score-wise Dynamic programming algorithm
    NOTE: For DP[v][l], v is 1-based and l is 1-based

    Input:
    -----
    val {numpy.ndarray}     : Value mapping vector
    freq {numpy.ndarray}    : Frequency vector (1-based)
    threshold               : Score threshold
    mode {'min' or 'max'}   : Configure to minimize or maximize the metric (Default: 'min')
    metric {'entropy', ...} : Chosen metric (Default: 'entropy')
    L                       : Maximum number of discretized bin (Default: 5) 
    R                       : Maximum number of discretized bin (Default: 15) 

    Output:
    -----
    dp_record {List}        : List of dp[n_val][l]; l \in [1, R]
    split_val {List}        : List of split values (upperbounds are used)
    opt_L {int}             : The number of discretized bin chosen
    
    '''
    print("Start of scoreDP")
    n_val = freq.shape[1] - 1

    dp = np.zeros(shape = (n_val + 10, R + 10)) # dp solution table
    trace = np.zeros(shape = (n_val + 10, R + 10)) - 1 # trace solutions
    dp[n_val][1] = metrics.getMetric(freq[:, 1:], freq, metric) # 1-bin solution
    
    opt_l = L # optimal number of bin, use for tracing
    opt_score = 0 # optimal score, use for calculating opt_l
    
    dp_record = [dp[n_val][1]] # Initialize dp_record with dp[n_val][1]
    
    # Initialize
    for v in range(1, n_val+1):
        dp[v, 1] = metrics.getMetric(freq[:, 1:v+1], freq, metric)
    
    # Score-wise DP
    for l in range(2, R+1):
        # Consider the l-th layer
        
        for v in range(l, n_val+1):
            # The bin ended with v
            dp[v, l] = utils.initMode(mode)

            for u in range(l-1, v):
                # The bin starts with u+1
                cand_dp = dp[u, l-1] + metrics.getMetric(freq[:, u+1:v+1], freq, metric)
                dp[v, l], update = Opt(dp[v, l], cand_dp, mode) # Update dp w.r.t mode (min/max)
                
                if update:
                    dp[v, l] = cand_dp
                    trace[v, l] = u # The bin (u, v] is included in the corresponding solution

        opt_score, update = Opt(opt_score, dp[n_val, l], mode) # Update opt_score
        if update:
            opt_l = l # Update opt_l

        print(f'{metric}: {dp[n_val, l]:.3f}, layer: {l}, mode: {mode}')
        dp_record.append(dp[n_val, l])

    # Rules tracing
    cur_bin = int(n_val)
    split_val = []
    cur_l = opt_l

    while cur_bin > 0:
        split_val.append(val[cur_bin-1]) # freq is 1-based while val is 0-based
        cur_bin = int(trace[cur_bin][cur_l])
        cur_l -= 1

    return opt_score, split_val, opt_l

def equalSize(df, FEATURE, n_bin, freq, val, metric = 'entropy'):
    '''
    Equal Size binning over df[FEATURE]
    '''
    if n_bin == len(val):
        score = 0.0
        for i in range(len(val)):
            score += metrics.getMetric(freq[:, i+1], freq, metric)
        return score

    fea = df[FEATURE].dropna()
    fea_val = np.array(fea)
    val_list = list(val)
    eS_mapping = np.array(pd.qcut(fea, q = n_bin, labels= False, duplicates= 'drop'))
    score = 0.0

    for i in range(n_bin):
        idx = []
        for j in range(len(fea_val)):
            if eS_mapping[j] == i:
                # idx.append(val_list.index(fea_val[j]) + 1)
                idx.append(utils.binSearch(val_list, fea_val[j]) + 1)
        idx = list(set(idx))
        if len(idx) > 0:
            score += metrics.getMetric(freq[:, idx], freq, metric)

    return score

def equalWidth(df, FEATURE, n_bin, freq, val, metric = 'entropy'):
    '''
    Equal Width binning over df[FEATURE]
    '''
    fea = df[FEATURE].dropna()
    fea_val = np.array(fea)
    val_list = list(val)
    eW_mapping = np.array(pd.cut(fea, bins = n_bin, labels= False))
    score = 0.0

    for i in range(n_bin):
        idx = []
        for j in range(len(fea_val)):
            if eW_mapping[j] == i:
                # idx.append(val_list.index(fea_val[j]) + 1)
                idx.append(utils.binSearch(val_list, fea_val[j]) + 1)
        idx = list(set(idx))
        if len(idx) > 0:
            score += metrics.getMetric(freq[:, idx], freq, metric)

    return score