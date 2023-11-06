import random
import numpy as np
'''
def PSO(objf, lb, ub, dim, PopSize, iters):

    # PSO parameters

    Vmax = 6
    wMax = 0.9
    wMin = 0.2
    c1 = 2
    c2 = 2

    s = solution()
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    ######################## Initializations'''

    

def iterarPSO(maxIter, t, dimension, poblacion, Best_P,pop_best):
    Vmax = 6
    wMax = 0.9
    wMin = 0.2
    c1 = 2
    c2 = 2

    vel = np.zeros((poblacion.__len__(), dimension))
    w = wMax - t * ((wMax - wMin) / maxIter)
    
    for i in range(0, poblacion.__len__()):
        for j in range(0, dimension):
            r1 = random.random()
            r2 = random.random()
            vel[i, j] = (
                w * vel[i, j]
                + c1 * r1 * (pop_best[i][j] - poblacion[i][j])
                + c2 * r2 * (Best_P[j] - poblacion[i][j])
            )

            if vel[i, j] > Vmax:
                vel[i, j] = Vmax

            if vel[i, j] < -Vmax:
                vel[i, j] = -Vmax
            poblacion[i][j] = poblacion[i][j] + vel[i][j]


    return np.array(poblacion)