import random
import math
import numpy as np
from numpy import linalg as LA
from cmath import inf, nan
from util import util

def iterarHBA(maxIter, t, dimension, poblacion, fitness, pob):
    C = 2
    alpha = C*math.exp(-t/maxIter)
    beta = 6
    vec_flag=[1,-1]
    vec_flag=np.array(vec_flag)
    posicionesOrdenadas = util.selectionSort(fitness)
    #print(poblacion[0,0])

    ########################Calculo de intensidad#########################################
    epsilon = 0.00000000000000022204
    di = np.zeros(pob)
    S = np.zeros(pob)
    I = np.zeros(pob)
    #fl=-10                    # The lower bound of the search interval.
    #ul=10                     # The upper bound of the search interval.
    poblacion = np.array(poblacion)
    #lb = fl*np.ones([dimension, 1])
    #ub = ul*np.ones([dimension, 1])

    for j in range(pob):
        if j < pob-1:
            di[j] = LA.norm([[poblacion[j,:]-poblacion[posicionesOrdenadas[0],:]+epsilon]]) #creo que está al revés, xprey - xi, el código en matlab esta así
            S[j]  = LA.norm([[poblacion[j,:]-poblacion[j+1,:]+epsilon]])
            di[j] = np.power(di[j], 2)
            S[j]  = np.power(S[j], 2)
        else:
            di[j] = LA.norm([[poblacion[pob-1,:]-poblacion[posicionesOrdenadas[0],:]+epsilon]])
            S[j]  = LA.norm([[poblacion[pob-1,:]-poblacion[1,:]+epsilon]])
            di[j] = np.power(di[j], 2)
            S[j]  = np.power(S[j], 2)

        if(di[j] == 0):
            print("di[i] igual a 0")
        n    = random.random()

        #print("S[",j,"]", S[j])
        #print("di[",j,"]", di[j])
        I[j] = n*S[j]/[4*math.pi*di[j]] #se ve bien
       
    
    #######################################################################################
        #Vs = random.random()
        F=vec_flag[math.floor((2*random.random()))] #revisar, pero se ve bien
        for k in range(dimension): #seguro que la actualización es po dimension y no individuo?
            Vs = random.random()
            di_number=poblacion[posicionesOrdenadas[0],k]-poblacion[j,k]
            if Vs < 0.5:
                r3=np.random.random()
                r4=np.random.randn()
                r5=np.random.randn()
                poblacion[j,k]= poblacion[posicionesOrdenadas[0],j] +F*beta*I[j]* poblacion[posicionesOrdenadas[0],k]+F*r3*alpha*(di_number)*np.abs(math.cos(2*math.pi*r4)*(1-math.cos(2*math.pi*r5)))
    
            else:
                r7 = random.random()
                poblacion[j,k] = poblacion[posicionesOrdenadas[0],j]+F*r7*alpha*di_number
             
        #poblacion[j,:] = BorderCheck1(poblacion[j,:],lb,ub,dimension)
        #PONER FUNCIÓN DE CORECCIÓN DE BORDES
        #print(poblacion)
        #print(poblacion)
        
        #if verificar(poblacion[j,:],dimension):
            #print("********************************************************")
            #print("NaN")
            #print("********************************************************")
        
        #print(poblacion[j,:])
        
    return np.array(poblacion) 

def BorderCheck1(X,lb,ub,dim):
    #print("CHEQUEa BORDE")
    for j in range(dim):
        if X[j]<lb[j]:
            X[j] = ub[j]
        elif X[j]>ub[j]:
            X[j] = lb[j]
    return X    

'''
def verificar(X,dim):
#funcion para ver si habían nan en los vectores
    for i in range(dim):
        if X[i] == nan:
            return True

    for i in range(dim):
        if X[i] == inf:
            return True

    return False'''