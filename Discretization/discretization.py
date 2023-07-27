import math
import random
import numpy as np 
from scipy import special as scyesp
from ChaoticMaps.chaoticMaps import logisticMap, piecewiseMap, sineMap, singerMap, sinusoidalMap, tentMap, circleMap

def aplicarBinarizacion(ind, transferFunction, binarizationFunction, bestSolutionBin, indBin, iter, iteraciones):
    individuoBin = []
    for i in range(ind.__len__()):
        individuoBin.append(0)

    for i in range(ind.__len__()):
        step1 = transferir(transferFunction, ind[i])
        individuoBin[i] = binarizar(binarizationFunction, step1, bestSolutionBin[i], indBin[i], iter, iteraciones)
    return np.array(individuoBin)

def transferir(transferFunction, dimension):
    if transferFunction == "S1":
        return S1(dimension)
    if transferFunction == "S2":
        return S2(dimension)
    if transferFunction == "S3":
        return S3(dimension)
    if transferFunction == "S4":
        return S4(dimension)
    if transferFunction == "V1":
        return V1(dimension)
    if transferFunction == "V2":
        return V2(dimension)
    if transferFunction == "V3":
        return V3(dimension)
    if transferFunction == "V4":
        return V4(dimension)
    if transferFunction == "X1":
        return X1(dimension)
    if transferFunction == "X2":
        return X2(dimension)
    if transferFunction == "X3":
        return X3(dimension)
    if transferFunction == "X4":
        return X4(dimension)
    if transferFunction == "Z1":
        return Z1(dimension)
    if transferFunction == "Z2":
        return Z2(dimension)
    if transferFunction == "Z3":
        return Z3(dimension)
    if transferFunction == "Z4":
        return Z4(dimension)


def binarizar(binarizationFunction, step1, bestSolutionBin, indBin, iter, iteraciones):
    if binarizationFunction == "STD":
        return Standard(step1)
    if binarizationFunction == "COM":
        return Complement(step1, indBin)
    if binarizationFunction == "PS":
        return ProblabilityStrategy(step1, indBin)
    if binarizationFunction == "ELIT":
        return Elitist(step1, bestSolutionBin)
    if binarizationFunction == "COM_LOG":
        return Complement_LOG(step1, indBin, iter, iteraciones)
    if binarizationFunction == "COM_PIECE":
        return Complement_PIECE(step1, indBin, iter, iteraciones)
    if binarizationFunction == "COM_SINE":
        return Complement_SINE(step1, indBin, iter, iteraciones)
    if binarizationFunction == "COM_SINGER":
        return Complement_SINGER(step1, indBin, iter, iteraciones)
    if binarizationFunction == "COM_SINU":
        return Complement_SINU(step1, indBin, iter, iteraciones)
    if binarizationFunction == "COM_TENT":
        return Complement_TENT(step1, indBin, iter, iteraciones)
    if binarizationFunction == "COM_CIRCLE":
        return Complement_CIRCLE(step1, indBin, iter, iteraciones)


def S1(dimension):
    return np.divide( 1 , ( 1 + np.exp( -2 * dimension ) ) )
def S2(dimension):
    return np.divide( 1 , ( 1 + np.exp( -1 * dimension ) ) )
def S3(dimension):
    return np.divide( 1 , ( 1 + np.exp( np.divide( ( -1 * dimension ) , 2 ) ) ) )
def S4(dimension):
    return np.divide( 1 , ( 1 + np.exp( np.divide( ( -1 * dimension ) , 3 ) ) ) )
def V1(dimension):
    return np.abs( scyesp.erf( np.divide( np.sqrt( np.pi ) , 2 ) * dimension ) )
def V2(dimension):
    return np.abs( np.tanh( dimension ) )
def V3(dimension):
    return np.abs( np.divide( dimension , np.sqrt( 1 + np.power( dimension , 2 ) ) ) )
def V4(dimension):
    return np.abs( np.divide( 2 , np.pi ) * np.arctan( np.divide( np.pi , 2 ) * dimension ) )
def X1(dimension):
    return np.divide( 1 , ( 1 + np.exp( 2 * dimension ) ) )
def X2(dimension):
    return np.divide( 1 , ( 1 + np.exp( dimension ) ) )
def X3(dimension):
    return np.divide( 1 , ( 1 + np.exp( np.divide( dimension , 2 ) ) ) )
def X4(dimension):
    return np.divide( 1 , ( 1 + np.exp( np.divide( dimension , 3 ) ) ) )
def Z1(dimension):
    return np.power( ( 1 - np.power( 2 , dimension ) ) , 0.5 )
def Z2(dimension):
    return np.power( ( 1 - np.power( 5 , dimension ) ) , 0.5 )
def Z3(dimension):
    return np.power( ( 1 - np.power( 8 , dimension ) ) , 0.5 )
def Z4(dimension):
    return np.power( ( 1 - np.power( 20 , dimension ) ) , 0.5 )


# def S1(dimension):
#     ans = 0
#     try:
#         ans = 1 / ( 1 + math.exp(-2*dimension) )
#     except OverflowError:
#         ans = float('inf')
#     return  ans
# def S2(dimension):
#     ans = 0
#     try:
#         ans = 1 / ( 1 + math.exp(-1*dimension) )
#     except OverflowError:
#         ans = float('inf')
#     return  ans
# def S3(dimension):
#     ans = 0
#     try:
#         ans = 1 / ( 1 + math.exp((-1*dimension) / 2 ) )
#     except OverflowError:
#         ans = float('inf')
#     return  ans
# def S4(dimension):
#     ans = 0
#     try:
#         ans = 1 / ( 1 + math.exp((-1*dimension)/3) ) 
#     except OverflowError:
#         ans = float('inf')
#     return  ans
# #def V1(dimension):
# #    return np.abs(scyesp.erf(np.divide(np.sqrt(math.pi),2)*dimension))
# def V2(dimension):
#     ans = 0
#     try:
#         ans = abs(math.tanh(dimension))
#     except OverflowError:
#         ans = float('inf')
#     return  ans
# def V3(dimension):
#     ans = 0
#     try:
#         ans = abs(dimension / math.sqrt(1+math.pow(dimension,2)))
#     except OverflowError:
#         ans = float('inf')
#     return  ans
# def V4(dimension):
#     ans = 0
#     try:
#         ans = abs(2 / math.pi)*math.atan((math.pi / 2)*dimension )
#     except OverflowError:
#         ans = float('inf')
#     return  ans

def Standard(step1):
    rand = random.uniform(0.0, 1.0)
    binario = 0
    if rand <= step1:
        binario = 1
    return binario

def Complement(step1, bin):
    rand = random.uniform(0.0, 1.0)
    binario = 0
    if rand <= step1:
        if bin == 1:
            binario = 0
        if bin == 0:
            
            binario =  1
    return binario

def Complement_LOG(step1, bin, iter, iteraciones):
    maps = logisticMap(0.7, iteraciones)
    rand = maps[iter]
    binario = 0
    if rand <= step1:
        if bin == 1:
            binario = 0
        if bin == 0:
            
            binario =  1
    return binario

def Complement_PIECE(step1, bin, iter, iteraciones):
    maps = piecewiseMap(0.7, iteraciones)
    rand = maps[iter]
    binario = 0
    if rand <= step1:
        if bin == 1:
            binario = 0
        if bin == 0:
            
            binario =  1
    return binario

def Complement_SINE(step1, bin, iter, iteraciones):
    maps = sineMap(0.7, iteraciones)
    rand = maps[iter]
    binario = 0
    if rand <= step1:
        if bin == 1:
            binario = 0
        if bin == 0:
            
            binario =  1
    return binario

def Complement_SINGER(step1, bin, iter, iteraciones):
    maps = singerMap(0.7, iteraciones)
    rand = maps[iter]
    binario = 0
    if rand <= step1:
        if bin == 1:
            binario = 0
        if bin == 0:
            
            binario =  1
    return binario

def Complement_SINU(step1, bin, iter, iteraciones):
    maps = sinusoidalMap(0.7, iteraciones)
    rand = maps[iter]
    binario = 0
    if rand <= step1:
        if bin == 1:
            binario = 0
        if bin == 0:
            
            binario =  1
    return binario

def Complement_TENT(step1, bin, iter, iteraciones):
    maps = tentMap(0.5, iteraciones)
    rand = maps[iter]
    binario = 0
    if rand <= step1:
        if bin == 1:
            binario = 0
        if bin == 0:
            
            binario =  1
    return binario

def Complement_CIRCLE(step1, bin, iter, iteraciones):
    maps = circleMap(0.7, iteraciones)
    rand = maps[iter]
    binario = 0
    if rand <= step1:
        if bin == 1:
            binario = 0
        if bin == 0:
            
            binario =  1
    return binario

def ProblabilityStrategy(step1, bin):
    alpha = 1/3
    binario = 0
    if alpha < step1 and step1 <= ( ( 1/2 ) * ( 1 + alpha ) ):
        binario = bin
    if step1 > ( ( 1/2 ) * ( 1 + alpha ) ):
        binario = 1
    return binario

def Elitist(step1, bestBin):
    rand = random.uniform(0.0, 1.0)
    binario = 0
    if rand < step1:
        binario = bestBin
    return binario
