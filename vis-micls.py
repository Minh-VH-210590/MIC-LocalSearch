# MIC-Local search algorithm is called in `mysubplot` procedure

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from binlib import mic

rs = np.random.RandomState(seed=0)

def mysubplot(x, y, numRows, numCols, plotNum,
              xlim=(-4, 4), ylim=(-4, 4)):

    r = np.around(np.corrcoef(x, y)[0, 1], 1)

    array = np.stack((x, y), axis= 1)
    df = pd.DataFrame(array, columns= ['X', 'Y'])
    print(len(df))
    print(df.head())

    ### MIC Local Search ###

    # Exhaustive initialization
    # Exhaustive search over fixed column X, where X is divided into 2 bins
    scoreListX, SX, archiveX = mic.genSolution(df, 'X', 'Y', exact= True, fixcol= 0)
    # Exhaustive search over fixed column Y, where X is divided into 2 bins
    scoreListY, SY, archiveY = mic.genSolution(df, 'Y', 'X', exact= True, fixcol= 1)
    # Take the best solution found
    SCORE = np.around(max(max(scoreListX), max(scoreListY)), 3)
    if max(scoreListX) > max(scoreListY):
        S0 = SX
    else:
        S0 = SY
    # List of visited solution
    init_archive = set(list(archiveX) + list(archiveY))
    
    # S0 = mic.genSolution(df, 'X', 'Y') # Generate random solution
    '''
    Find solution regarding columns named 'X' and 'Y' from initial solution S0
    over T = 20 iterations, with nseed = 30 neighbor solutions considered at each iteration.
    To avoid revisiting solutions, init_archive is added.
    '''
    S = mic.MIC_LocalSearch(df, 'X', 'Y', T= 20, S= S0, nseed= 30, init_archive= init_archive)
    SCORE = np.around(S.score, 3)

    ax = plt.subplot(numRows, numCols, plotNum,
                     xlim=xlim, ylim=ylim)
    ax.set_title('Pearson r=%.1f\nMIC=%.3f' % (r, SCORE),fontsize=10)
    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.plot(x, y, '.')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def rotation(xy, t):
    return np.dot(xy, [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

def mvnormal(n=1000):
    cors = [1.0, 0.8, 0.4, 0.0, -0.4, -0.8, -1.0]
    # cors = [0.4, 0.0, -0.8]
    for i, cor in enumerate(cors):
        cov = [[1, cor],[cor, 1]]
        xy = rs.multivariate_normal([0, 0], cov, n)
        mysubplot(xy[:, 0], xy[:, 1], 3, 7, i+1)

def rotnormal(n=1000):
    ts = [0, np.pi/12, np.pi/6, np.pi/4, np.pi/2-np.pi/6,
          np.pi/2-np.pi/12, np.pi/2]
    # ts = [0, np.pi/12, np.pi/6, np.pi/2]
    cov = [[1, 1],[1, 1]]
    xy = rs.multivariate_normal([0, 0], cov, n)
    for i, t in enumerate(ts):
        xy_r = rotation(xy, t)
        mysubplot(xy_r[:, 0], xy_r[:, 1], 3, 7, i+8)

def others(n=1000):
    x = rs.uniform(-1, 1, n)
    y = 4*(x**2-0.5)**2 + rs.uniform(-1, 1, n)/3
    mysubplot(x, y, 3, 7, 15, (-1, 1), (-1/3, 1+1/3))

    y = rs.uniform(-1, 1, n)
    xy = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    xy = rotation(xy, -np.pi/8)
    lim = np.sqrt(2+np.sqrt(2)) / np.sqrt(2)
    mysubplot(xy[:, 0], xy[:, 1], 3, 7, 16, (-lim, lim), (-lim, lim))

    xy = rotation(xy, -np.pi/8)
    lim = np.sqrt(2)
    mysubplot(xy[:, 0], xy[:, 1], 3, 7, 17, (-lim, lim), (-lim, lim))

    y = 2*x**2 + rs.uniform(-1, 1, n)
    mysubplot(x, y, 3, 7, 18, (-1, 1), (-1, 3))

    y = (x**2 + rs.uniform(0, 0.5, n)) * \
        np.array([-1, 1])[rs.random_integers(0, 1, size=n)]
    mysubplot(x, y, 3, 7, 19, (-1.5, 1.5), (-1.5, 1.5))

    y = np.cos(x * np.pi) + rs.uniform(0, 1/8, n)
    x = np.sin(x * np.pi) + rs.uniform(0, 1/8, n)
    mysubplot(x, y, 3, 7, 20, (-1.5, 1.5), (-1.5, 1.5))

    xy1 = rs.multivariate_normal([3, 3], [[1, 0], [0, 1]], int(n/4))
    xy2 = rs.multivariate_normal([-3, 3], [[1, 0], [0, 1]], int(n/4))
    xy3 = rs.multivariate_normal([-3, -3], [[1, 0], [0, 1]], int(n/4))
    xy4 = rs.multivariate_normal([3, -3], [[1, 0], [0, 1]], int(n/4))
    xy = np.concatenate((xy1, xy2, xy3, xy4), axis=0)
    mysubplot(xy[:, 0], xy[:, 1], 3, 7, 21, (-7, 7), (-7, 7))

plt.figure(facecolor='white')
mvnormal(n=150)
rotnormal(n=100)
others(n=150)
plt.tight_layout()
plt.show()