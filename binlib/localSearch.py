# Utilities for local search
import numpy as np

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

def Mutate(S, p, maxD):
    '''
    Mutate current solution S, using the probability vector p for operators

    p[0]: p_fix
    p[1]: p_add
    p[2]: p_remove
    p[3]: p_move
    '''
    nS = Solution(S.maskX, S.maskY, S.fixCol, S.score)

    # Switch fixed column
    if np.random.binomial(1, p[0]) == 1:
        nS.fixCol = 1 - nS.fixCol
    
    mask = nS.maskY
    if nS.fixCol == 0:
        mask = nS.maskX

    # Mutate operations
    if np.random.binomial(1, p[1]) == 1:
        turnOn(mask)
    if np.random.binomial(1, p[2]) == 1:
        turnOff(mask)
    if np.random.binomial(1, p[3]) == 1:
        localMove(mask, maxD)

    if nS.fixCol == 0:
        nS.maskX = mask
    else:
        nS.maskY = mask

    return nS