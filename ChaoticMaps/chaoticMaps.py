import numpy as np
import matplotlib.pyplot as plt
import math

def chebyshevMap(initial,iteration):
    map_values = np.zeros(iteration)
       
    
    i = 1
    map_values[0] = initial
    x_previo = initial
    while i < iteration:
        
        x = math.cos( i * ( 1 / math.cos( x_previo ) ) )
        
        map_values[i] = x
        
        x_previo = x
        
        i+=1
    return map_values

def gaussianAndGauss_mouseMap(initial,iteration):
    map_values = np.zeros(iteration)
       
    
    i = 1
    map_values[0] = initial
    x_previo = initial
    while i < iteration:
        
        if x_previo == 0:
            x = 0
        else:
            x = ( 1 / ( x_previo % 1.0 ) )
        
        map_values[i] = x
        
        x_previo = x
        
        i+=1
    return map_values

def circleMap(initial,iteration):
    map_values = np.zeros(iteration)
    a = 0.5
    b = 0.2    
    k = a / ( 2 * math.pi )
    
    
    i = 1
    map_values[0] = initial
    x_previo = initial
    while i < iteration:
        
        x = ( x_previo + b - k * math.sin( 2 * math.pi * x_previo ) ) % 1.0
        
        map_values[i] = x
        
        x_previo = x
        
        i+=1
    return map_values

def logisticMap(initial,iteration):
    map_values = np.zeros(iteration)
    a = 4
    i = 1
    map_values[0] = initial
    x_previo = initial
    while i < iteration:
        
        x = a * x_previo * (1 - x_previo)
        
        map_values[i] = x
        
        x_previo = x
        
        i+=1
    return map_values
    
def piecewiseMap(initial,iteration):
    map_values = np.zeros(iteration)
    P = 0.4
    i = 1
    map_values[0] = initial
    x_previo = initial
    while i < iteration:
        
        if P > x_previo and x_previo >= 0:
            x = x_previo / P
        if 1/2 > x_previo and x_previo >= P:
            x = ( x_previo - P ) / ( 0.5 - P )
        if (1-P) > x_previo and x_previo >= 1/2:
            x = ( 1 - P - x_previo ) / ( 0.5 - P )
        if 1 > x_previo and x_previo >= ( 1 - P ):
            x = ( 1 - x_previo ) / P
        
            
        
        map_values[i] = x
        
        x_previo = x
        
        i+=1
    return map_values

def sineMap(initial,iteration):
    map_values = np.zeros(iteration)
    a = 4
    i = 1
    map_values[0] = initial
    x_previo = initial
    while i < iteration:
        
        x = ( a / 4 ) * ( math.sin( math.pi * x_previo ) )
        
        map_values[i] = x
        
        x_previo = x
        
        i+=1
    return map_values

def singerMap(initial,iteration):
    map_values = np.zeros(iteration)
    a = 4
    u = 1.07
    i = 1
    map_values[0] = initial
    x_previo = initial
    while i < iteration:
        
        x = u * ( ( 7.86 * x_previo ) - ( 23.31 * pow(x_previo,2) ) + ( 28.75 * pow(x_previo,3) ) - ( 13.302875 * pow(x_previo,4) ) )
        
        map_values[i] = x
        
        x_previo = x
        
        i+=1
    return map_values

def sinusoidalMap(initial,iteration):
    map_values = np.zeros(iteration)
    a = 2.3
    i = 1
    map_values[0] = initial
    x_previo = initial
    while i < iteration:
        
        x = a * pow(x_previo,2) * math.sin( math.pi * x_previo )
        
        map_values[i] = x
        
        x_previo = x
        
        i+=1
    return map_values

def tentMap(initial,iteration):
    map_values = np.zeros(iteration)
    
    i = 1
    map_values[0] = initial
    x_previo = initial
    while i < iteration:
        
        if x_previo < 0.7:
            x = x_previo / 0.7
        else:
            x = ( 10 / 3 ) * ( 1 - x_previo )
        
        map_values[i] = x
        
        x_previo = x
        
        i+=1
    return map_values


def graficar(iteration):
    iterationes = np.zeros(iteration)
    
    for i in range(iteration):
        iterationes[i] = i + 1
    
    grafico = 1
    
    while grafico < 8:
        
        if grafico == 1:
            plt.plot(iterationes, logisticMap(0.7,iteration), label="logistic Map", marker="*")
            plt.title("logistic Map")
            plt.xlim(0,iteration)
            plt.ylim(0,1)
        if grafico == 2:
            plt.plot(iterationes, piecewiseMap(0.7,iteration), label="piecewise Map", marker="*")
            plt.title("piecewise Map")
            plt.xlim(0,iteration)
            plt.ylim(0,1)
        if grafico == 3:
            plt.plot(iterationes, sineMap(0.7,iteration), label="sine Map", marker="*")
            plt.title("sine Map")
            plt.xlim(0,iteration)
            plt.ylim(0,1)
        if grafico == 4:
            plt.plot(iterationes, singerMap(0.7,iteration), label="singer Map", marker="*")
            plt.title("singer Map")
            plt.xlim(0,iteration)
            plt.ylim(0,1)
        if grafico == 5:
            plt.plot(iterationes, sinusoidalMap(0.7,iteration), label="sinusoidal Map", marker="*")
            plt.title("sinusoidal Map")
            plt.xlim(0,iteration)
            plt.ylim(0,1)
        if grafico == 6:
            plt.plot(iterationes, tentMap(0.5,iteration), label="tent Map", marker="*")
            plt.title("tent Map")
            plt.xlim(0,iteration)
            plt.ylim(0,1)
        if grafico == 7:
            plt.plot(iterationes, circleMap(0.7,iteration), label="circle Map", marker="*")
            plt.title("circle Map")
            plt.xlim(0,iteration)
            plt.ylim(0,1)
        # if grafico == 8:
        #     plt.plot(iterationes, chebyshevMap(0.7,iteration), label="chebyshev Map", marker="*")
        #     plt.title("chebyshev Map")
        #     plt.xlim(0,iteration)
        #     plt.ylim(-1,1)
        # if grafico == 9:
        #     plt.plot(iterationes, gaussianAndGauss_mouseMap(0.7,iteration), label="gaussian and Gauss-mouse Map", marker="*")
        #     plt.title("gaussian and Gauss-mouse Map")
        #     plt.xlim(0,iteration)
        #     plt.ylim(0,1)
        # plt.xlabel('$iterations (k)$')
        # plt.ylabel('$Value (x_{k})$')
        # plt.legend(loc="upper right")
        
        
        
        

        
        
        
        # plt.grid()
        # plt.savefig("Funciones de Transferencia tipo X.pdf")
        plt.show()
        
        grafico+=1



graficar(50)