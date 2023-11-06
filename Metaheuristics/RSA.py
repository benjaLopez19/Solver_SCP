import random
import numpy as np

def iterarRSA(maxIter, t, dimension, poblacion, Best_P,LB,UB):
    #PARAM
    alfa = 0.1
    beta = 0.1
    #Small value epsilon
    eps = 1e-10
    
    #UPDATE ES
    r3 = random.randint(-1, 1) #r3 denotes to a random integer number between −1 and 1, pag4
    ES = 2*r3*(1-(1/maxIter))

    #Pob size
    N = poblacion.__len__()

    #ITER
    for i in range(N):
        for j in range(dimension):
            r2 = random.randint(0, N-1)
            R =  (Best_P[j] - poblacion[r2][j])/(Best_P[j]+eps)
            P = alfa + (poblacion[i][j]-np.mean(poblacion[i])) / (UB-LB+eps)
            Eta = Best_P[j]*P
            rand = random.random()
            ##Ecc de movimiento##

            #ec1
            if(t<maxIter/4):
                poblacion[i][j] = Best_P[j] - Eta*beta - R*rand
            #ec2
            elif(t<(2*maxIter)/4 and t>=maxIter/4):
                r1 = random.randint(0, N-1)
                poblacion[i][j] = Best_P[j] * poblacion[r1][j] * ES * rand
            #ec3
            elif(t<(maxIter*3)/4 and t>=(2*t)/4):
                poblacion[i][j] = Best_P[j] * P * rand
            #ec4
            else:
                poblacion[i][j] = Best_P[j] - Eta*eps - R*rand
        #Fin for dimension

    return np.array(poblacion)