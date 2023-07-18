import math
import os
import random
import numpy as np
from Problem.EMPATIA.model.fs_simple_methods import (create_filter_list, create_LASSO_foreach,
                                                    create_LASSO_list, create_wrapper_list)
# # lectura de datos EMPATIA
from Problem.EMPATIA.database.prepare_dataset import prepare_47vol_solap,  prepare_47vol_nosolap,  prepare_100vol_solap,  prepare_100vol_nosolap

# lectura modelo KNN 
from Problem.EMPATIA.model.ml_model import get_metrics
from Problem.EMPATIA.model.hyperparameter_optimization import load_parameters

def esDecimal(numero):
    try:
        float(numero)
        return True
    except:
        return False

def distEuclidiana(x, y, missd, missdValue):
    suma = 0
    for i in range(x.__len__()):
        if missd:
            if x[i] != missdValue and y[i] != missdValue:
                suma = suma + ( (x[i] - y[i])**2 )
        else:
            suma = suma + ((x[i] - y[i]) ** 2)
    return math.sqrt(suma)

def porcentajesXLPXPT(div, maxDiv):
    XPL = round((div/maxDiv)*100,2)
    XPT = round((abs(div-maxDiv)/maxDiv)*100,2)
    state = -1
    #Determinar estado
    if XPL >= XPT:
        state = 1 # Exploración
    else:
        state = 0 # Explotación
    return XPL, XPT, state

def generacionMixtaFS(poblacion, caracteristicas):

    pop = np.zeros(shape=(poblacion,caracteristicas))

    mayor = int(caracteristicas*0.8)
    menor = int(caracteristicas*0.3)



    individuo = 0
    for ind in pop:
        if individuo < int( len(pop) / 2) :
            L=[random.randint(0, caracteristicas-1)] #este es L[0]
            i=1
            while i<mayor:
                x=random.randint(0,caracteristicas-1)
                if x not in L:
                    L.append(x)
                    i+=1
            unos = sorted(L)
            individuo += 1
        else:
            L=[random.randint(0, caracteristicas-1)] #este es L[0]
            i=1
            while i<menor:
                x=random.randint(0,caracteristicas-1)
                if x not in L:
                    L.append(x)
                    i+=1

            unos = sorted(L)

        
        ind[unos] = 1
    return pop

def diversidadHussain(matriz):
    # [ [1,2,3,4,5,6],
    #   [6,5,4,3,2,1],
    #   [1,2,3,4,5,6],
    #   [6,5,4,3,2,1],
    #   [1,2,3,4,5,6],
    #   [6,5,4,3,2,1],
    #   [1,2,3,4,5,6] ]
    medianas = []
    for j in range(matriz[0].__len__()):
        suma = 0
        for i in range(matriz.__len__()):
            suma += matriz[i][j]
        medianas.append(suma/matriz.__len__())
    n = len(matriz)
    l = len(matriz[0])
    diversidad = 0
    for d in range(l):
        div_d = 0
        for i in range(n):
            div_d = div_d + abs(medianas[d] - matriz[i][d])
        diversidad = diversidad + div_d
    return (1 / (l*n)) * diversidad

def selectionSort(lista):
    posiciones = []
    for i in range(len(lista)):
        posiciones.append(i) 
    for i in range(len(lista)):
        lowest_value_index = i
        for j in range(i + 1, len(lista)):
            if lista[j] < lista[lowest_value_index]:
                lowest_value_index = j
        lista[i], lista[lowest_value_index] = lista[lowest_value_index], lista[i]
        posiciones[i], posiciones[lowest_value_index] = posiciones[lowest_value_index], posiciones[i]
    return posiciones

def normr(Mat):
    norma = 0
    for i in range(Mat.__len__()):
        norma = norma + abs(math.pow(Mat[i],2))
    norma = math.sqrt(norma)
    B = []
    for i in range(Mat.__len__()):
        B.append(Mat[i]/norma)
    return B

def getUbLb(poblacion, dimension):
    ub = []
    lb = []
    for j in range(dimension):
        lista = []
        for i in range(poblacion.__len__()):
            lista.append(poblacion[i][j])
        ordenLista = selectionSort(lista)
        ub.append(poblacion[ordenLista[poblacion.__len__()-1]][j])
        lb.append(poblacion[ordenLista[0]][j])    
    return ub, lb

def RouletteWheelSelection(weights):
    accumulation = sum(weights)
    p = random.random() * accumulation
    chosen_index = -1
    suma = 0
    for index in range(len(weights)):
        suma = suma + weights[index]
        if suma > p:
            chosen_index = index
            break
    choice = chosen_index
    return choice

# Create a function that converts a digital file into binary
def convert_into_binary(file_path):
    with open(file_path, 'rb') as file:
        binary = file.read()

    return binary

def writeTofile(data, filename):
    # Convert binary data to proper format and write it on Hard Disk
    with open(filename, 'wb') as file:
        file.write(data)
        
def invertirArray(vector):
    return vector[::-1]

def totalFeature():
    return 57

def get_fitness(loader, individuo, problema, opt):
    # Return the confusion matrix (TN, FP, FN, TP)
    
    if problema == 'EMPATIA' or problema == 'EMPATIA-2' or problema == 'EMPATIA-5' or problema == 'EMPATIA-6':
        scores = get_metrics(loader, 
                            neighbors = 20,
                            missclassification_cost = 1.6,
                            selected_features=individuo)
    if problema == 'EMPATIA-3' or problema == 'EMPATIA-4' or problema == 'EMPATIA-7' or problema == 'EMPATIA-8' or problema == 'EMPATIA-9' or problema == 'EMPATIA-10' or problema == 'EMPATIA-11' or problema == 'EMPATIA-12':
        scores = get_metrics(loader, 
                            selected_features=individuo,
                            optimal_parameters=opt,
                            threshold = 0.126)
    
    tn = scores[0]
    fp = scores[1]
    fn = scores[2]
    tp = scores[3]
    alpha_1 = 0.99
    alpha_2 = 1 -alpha_1
    
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    errorRate   = 1 - accuracy
    mcc         = ( (tp * tn) - (fp * fn) ) / ( np.sqrt( (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) ) )

    precision   = tp / (tp + fp)
    recall      = tp / (tp + fn)
    f1          = 2 * ( (precision * recall) / (precision + recall) )
    
    if problema == 'EMPATIA' or problema == 'EMPATIA-3' or problema == 'EMPATIA-5' or problema == 'EMPATIA-7':
        fitness = (alpha_1 * errorRate) + (alpha_2 * (sum(individuo)/totalFeature()))
    if problema == 'EMPATIA-2' or problema == 'EMPATIA-4' or problema == 'EMPATIA-6' or problema == 'EMPATIA-8' or problema == 'EMPATIA-9' or problema == 'EMPATIA-10' or problema == 'EMPATIA-11' or problema == 'EMPATIA-12':
        fitness = 1 - f1
    
    return np.round(fitness,3), np.round(accuracy,3), np.round(f1,3), np.round(precision,3), np.round(recall,3), np.round(mcc,3), np.round(errorRate,3), sum(individuo)


def generate_population_filter_methods_empatia(instancia,pop):

    data_dir = './Problem/EMPATIA/'

    opt = None
    if instancia == 'EMPATIA-9':
        opt = load_parameters(data_dir, "optimal_parameters_47sol")
    if instancia == 'EMPATIA-10':
        opt = load_parameters(data_dir, "optimal_parameters_47nosol")
    if instancia == 'EMPATIA-11':
        opt = load_parameters(data_dir, "optimal_parameters_100sol")
    if instancia == 'EMPATIA-12':
        opt = load_parameters(data_dir, "optimal_parameters_100nosol")

    problema = instancia

    # LOAD DATA

    loader = None
    # según el dataset que quieras cargar tendrás que llamar a las diferentes funciones 

    # 47vol_solap : 47 voluntarias con solapamiento
    if instancia == 'EMPATIA-9':
        loader = prepare_47vol_solap(data_dir)

    # 47vol_nosolap: 47 voluntarias sin solapamiento
    if instancia == 'EMPATIA-10':
        loader = prepare_47vol_nosolap(data_dir)

    # 100vol_solap: 100 voluntarias con solapamiento
    if instancia == 'EMPATIA-11':
        loader = prepare_100vol_solap(data_dir)

    # 100vol_nosolap: 100 voluntarias sin solapamiento
    if instancia == 'EMPATIA-12':
        loader = prepare_100vol_nosolap(data_dir)

    # voluntarias a eliminar de las 47
    if instancia == 'EMPATIA-9' or instancia == 'EMPATIA-10':
        loader.dataset = loader.exclude_volunteers([11,13,31,62,83])

    # voluntarias a eliminar de las 100
    if instancia == 'EMPATIA-11' or instancia == 'EMPATIA-12':
        loader.dataset = loader.exclude_volunteers([12,65,67,69,82,83,92,94,123,134,136,139]) 
        
    # seleccion = create_wrapper_list(loader = loader, n_variables = 10, label_to_predict = 'y')

    # fitness, accuracy, f1Score, presicion, recall, mcc, errorRate, totalFeatureSelected = get_fitness(loader, seleccion, problema, opt)

    # print(seleccion)
    # print(f'''{accuracy},{f1Score},{presicion},{recall},{errorRate},{mcc},{totalFeatureSelected}''')

    population = np.zeros(shape=(pop,totalFeature()))
    fitnesss = []
    solutions = []
    tfss = []
    
    th = 0.001

    while th < 0.05:

        seleccion = create_filter_list(loader = loader, threshold = th, label_to_predict = 'y')

        if np.sum(seleccion) > 0:
        
            fitness, accuracy, f1Score, presicion, recall, mcc, errorRate, totalFeatureSelected = get_fitness(loader, seleccion, problema, opt)
            fitnesss.append(f1Score)
            solutions.append(seleccion)
            tfss.append(totalFeatureSelected)    

        th = np.round(th+0.0005,5)
    
    print(tfss)