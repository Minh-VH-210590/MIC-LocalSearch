# Utilities for local search
import numpy as np

from binlib import mic, utils, metrics

# Solution Class #
class Solution:
    def __init__(self, maskX, maskY, fixCol, score):
        self.maskX = maskX
        self.maskY = maskY
        self.fixCol = fixCol
        self.score = score

    def _encode(self):
        '''
        Encode a solution for archive purposes (tabu search, ...)
        '''
        enc_str = str(self.fixCol)
        for i in range(self.maskX.shape[0]):
            enc_str = enc_str + str(self.maskX[i])
        for i in range(self.maskY.shape[0]):
            enc_str = enc_str + str(self.maskY[i])
        return enc_str
    
    def _bit_cnt(self, mask):
        '''
        Count the number of active bit over mask
        '''
        cnt = 0
        for i in range(len(mask)):
            if mask[i] == 1:
                cnt += 1
        return cnt
    
    def _export(self, out_dir, extras= None):
        '''
        Export solution
        '''
        with open(out_dir, 'a') as file:
            file.write(f'Final MIC: {self.score}\n')
            file.write(f'Feature #1 is discretized into {self._bit_cnt(self.maskX) + 1} bins.\n')
            file.write(f'Feature #2 is discretized into {self._bit_cnt(self.maskY) + 1} bins.\n')
            if extras is not None:
                for extra in extras:
                    file.write(str(extra) + '\n')
            file.close()

    def _copy(self):
        nS = Solution(self.maskX, self.maskY, self.fixCol, self.score)
        return nS
    
    def _max_score(self, df, X, Y, valX, valY, metric= 'mi', Bn= None):
        '''
        Calculate the maximum score obtained by the solution, 
        where the non-fixed column is not discretized
        '''
        score = 0
        if Bn is None:
            Bn = len(df) ** 0.6

        # X is fixed
        if self.fixCol == 0:
            dX_val = utils.discretizeFea(df, X, utils.mask2Split(self.maskX, valX))
            nX = max(dX_val) + 1
            nY = int(Bn / nX)

            if nX <= 1 or nY <= 1:
                return 0
            
            oldX = df[X].copy()
            df[X] = dX_val # Replace df[X] by its discretized version

            val, freq, _ = utils.makePrebins(df, Y, X, num_classes= nX)
            for i in range(len(val) + 1):
                score += metrics.getMetric(freq[:, i], freq, metric)

            df[X] = oldX.copy()
            # score = score / np.log2(min(nX, nY))
        
        else:
            dY_val = utils.discretizeFea(df, Y, utils.mask2Split(self.maskY, valY))
            nY = max(dY_val) + 1
            nX = int(Bn / nY)

            if nY <= 1 or nX <= 1:
                return 0
            
            oldY = df[Y].copy()
            df[Y] = dY_val # Replace df[X] by its discretized version

            val, freq, _ = utils.makePrebins(df, X, Y, num_classes= nY)
            for i in range(len(val) + 1):
                score += metrics.getMetric(freq[:, i], freq, metric)

            df[Y] = oldY.copy()
            # score = score / np.log2(min(nX, nY))

        # print(score)
        # return score
        return score

# Operators #
def turnOn(mask):
    '''
    Randomly turn on ONE bitmask.

    Output:
    -----
    mask {np.ndarray}   : Mask after modification
    '''
    n = mask.shape[0]
    i = np.random.randint(0, n)
    patience = 10
    while mask[i] != 0:
        i = np.random.randint(0, n)
        patience -= 1
        if patience == 0:
            break
    mask[i] = 1
    return mask

def exhaustTurnOn(mask):
    '''
    Turn on ONE bitmask.

    Output:
    -----
    mask_list {List}   : List of masks after modification
    '''
    n = mask.shape[0]
    S = Solution(mask, mask, 1, 0)
    if S._bit_cnt(mask) * 2 >= n ** 0.6:
        return []
    mask_list = []
    for i in range(n):
        if mask[i] == 0:
            mask[i] = 1
            mask_list.append(np.copy(mask))
            mask[i] = 0
    return mask_list

def turnOff(mask):
    '''
    Randomly turn off ONE bitmask.

    Output:
    -----
    mask {np.ndarray}   : Mask after modification
    '''
    n = mask.shape[0]
    i = np.random.randint(0, n)
    patience = 10
    while mask[i] != 1:
        i = np.random.randint(0, n)
        patience -= 1
        if patience == 0:
            break
    mask[i] = 0
    return mask

def exhaustTurnOff(mask):
    '''
    Turn off ONE bitmask.

    Output:
    -----
    mask_list {List}   : List of masks after modification
    '''
    n = mask.shape[0]
    S = Solution(mask, mask, 1, 0)
    if S._bit_cnt(mask) <= 1:
        return []
    mask_list = []
    for i in range(n):
        if mask[i] == 1:
            mask[i] = 0
            mask_list.append(np.copy(mask))
            mask[i] = 1
    return mask_list

def localMove(mask, maxD= 3):
    '''
    Randomly move one split points AT MOST maxD units to the right or left along the value axis

    Input:
    -----
    mask {np.ndarray}   : Original mask
    maxD {int}          : Maximum distance. Default = 3

    Output:
    -----
    mask {np.ndarray}   : Mask after modification
    '''
    # Choose split point to be moved. This split point must have mask = 1
    n = mask.shape[0]
    nLoop = 10
    while nLoop > 0:
        patience = 10
        nLoop -= 1

        # Choose bitmask to move
        i = np.random.randint(0, n)
        while mask[i] != 1:
            i = np.random.randint(0, n)
            patience -= 1
            if patience == 0:
                break

        if patience == 0:
            break
        
        patience = 6
        
        # Choose moving distance
        d = np.random.randint(1, maxD + 1)
        if np.random.randint(0, 2) == 1:
            d = -d
        di = min(i+d, n-1)
        di = max(di, 0)

        while mask[di] == 1:
            d = np.random.randint(1, maxD + 1)
            if np.random.randint(0, 2) == 1:
                d = -d
            di = min(i+d, n-1)
            di = max(di, 0)
            
            patience -= 1
            if patience == 0:
                break
        
        # If there are a moving scheme for current (i, maxD), conclude the new mask
        if patience != 0:
            mask[i] = 0 # Move the split point from i to di
            mask[di] = 1
            break
    
    return mask

def exhaustMove(mask, pos):
    '''
    Move one split point at position 'pos' of mask. 

    Input:
    -----
    mask {np.ndarray}   : Original mask

    Output:
    -----
    mask_list {List}   : List of mask after modification
    '''
    if mask[pos] == 0:
        return []
    mask_list = []
    n = mask.shape[0]
    
    mask[pos] = 0

    ptr = pos - 1
    while ptr >= 0 and mask[ptr] == 0:
        mask[ptr] = 1
        mask_list.append(np.copy(mask))
        mask[ptr] = 0
        ptr -= 1

    ptr = pos + 1
    while ptr < n and mask[ptr] == 0:
        mask[ptr] = 1
        mask_list.append(np.copy(mask))
        mask[ptr] = 0
        ptr += 1

    return mask_list


def Mutate(S, p, maxD, Bn):
    '''
    Mutate current solution S, using the probability vector p for operators

    p[0]: p_fix
    p[1]: p_add
    p[2]: p_remove
    p[3]: p_move
    '''
    nS = Solution(S.maskX, S.maskY, S.fixCol, S.score)
    
    if np.random.binomial(1, p[0]) == 1:
        nS.fixCol = 1 - nS.fixCol
        return [nS] # Change fixCol without any further mutation
    
    # Set fixCol
    mask = nS.maskY
    if nS.fixCol == 0:
        mask = nS.maskX
    nMask = nS._bit_cnt(mask)

    incFlag = True
    if 2 * (nMask + 1) >= Bn:
        incFlag = False

    decFlag = True
    if nMask <= 1:
        decFlag = False

    # Random mutation
    if np.random.binomial(1, p[3]) == 1 or ((not incFlag) and (not decFlag)):
        mask = localMove(mask, maxD)
    else:
        if np.random.binomial(1, p[1]) == 1 and incFlag:
            mask = turnOn(mask)
        else:
            mask = turnOff(mask)
            # if np.random.binomial(1, p[2]) == 1:
            #     mask = turnOff(mask)

    if nS.fixCol == 0:
        nS.maskX = mask
    else:
        nS.maskY = mask

    return [nS]

def exhaustMutate(S:Solution, df, X, Y, Bn):
    '''
    Exhaustively mutate current solution S, using the probability vector p for operators.

    Assuming X-axis is mutated
    '''
    candidates = []
    SmaskX = S.maskX
    SmaskY = S.maskY
    SfixCol = S.fixCol
    Sscore = S.score

    mask = SmaskX
    if S.fixCol == 1:
        mask = SmaskY

    valX = utils.makeVal(df, X)
    valY = utils.makeVal(df, Y)

    # TurnOn
    masks = exhaustTurnOn(mask)
    if S.fixCol == 0:
        for m in masks:
            nS = Solution(m, SmaskY, SfixCol, Sscore)
            if nS._max_score(df, X, Y, valX, valY, Bn= Bn) < Sscore:
                continue
            candidates.append((nS, 'turnOn'))
    else:
        for m in masks:
            nS = Solution(SmaskX, m, SfixCol, Sscore)
            if nS._max_score(df, X, Y, valX, valY, Bn= Bn) < Sscore:
                continue
            candidates.append((nS, 'turnOn'))

    # TurnOff
    masks = exhaustTurnOff(mask)
    if S.fixCol == 0:
        for m in masks:
            nS = Solution(m, SmaskY, SfixCol, Sscore)
            if nS._max_score(df, X, Y, valX, valY, Bn= Bn) < Sscore:
                continue
            candidates.append((nS, 'turnOff'))
    else:
        for m in masks:
            nS = Solution(SmaskX, m, SfixCol, Sscore)
            if nS._max_score(df, X, Y, valX, valY, Bn= Bn) < Sscore:
                continue
            candidates.append((nS, 'turnOff'))

    # Move
    n = len(mask)
    masks = []
    for i in range(n):
        if mask[i] == 1:
            masks = masks + exhaustMove(mask, i)
    
    if S.fixCol == 0:
        for m in masks:
            nS = Solution(m, SmaskY, SfixCol, Sscore)
            if nS._max_score(df, X, Y, valX, valY, Bn= Bn) < Sscore:
                continue
            candidates.append((nS, 'Move'))
    else:
        for m in masks:
            nS = Solution(SmaskX, m, SfixCol, Sscore)
            if nS._max_score(df, X, Y, valX, valY, Bn= Bn) < Sscore:
                continue
            candidates.append((nS, 'Move'))

    # Adding axis-switch solution by the end of candidates
    # candidates.append((Solution(SmaskX, SmaskY, 1 - SfixCol, Sscore), 'Switch'))
    
    return candidates

def Switch(S:Solution, df, X, Y):
    MAX_SCORE = S.score
    valX = utils.makeVal(df, X)
    valY = utils.makeVal(df, Y)
    
    while True:
        nS = S._copy()
        nS.fixCol = 1 - nS.fixCol # Switch axis

        if nS.fixCol == 0:
            spltX = utils.mask2Split(nS.maskX, valX)
            score, splt = mic.Opt_YbyX(df, X, Y, spltX)
            nS.score = score
            nS.maskY = utils.split2Mask(splt, valY)
        else:
            spltY = utils.mask2Split(nS.maskY, valY)
            score, splt = mic.Opt_YbyX(df, Y, X, spltY)
            nS.score = score
            nS.maskX = utils.split2Mask(splt, valX)
        
        if nS.score > MAX_SCORE:
            MAX_SCORE = nS.score
            S = nS._copy()
        else:
            return S