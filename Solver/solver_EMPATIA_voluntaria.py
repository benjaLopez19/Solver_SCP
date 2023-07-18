import numpy as np
import os
from Metaheuristics.GWO import iterarGWO 
from Metaheuristics.SCA import iterarSCA
from Metaheuristics.WOA import iterarWOA
from Metaheuristics.PSA import iterarPSA
from Metaheuristics.MFO import iterarMFO
from Metaheuristics.GA import iterarGA
from Diversity.hussainDiversity import diversidadHussain
from Diversity.XPLXTP import porcentajesXLPXPT
import time
from Discretization import discretization as b
from util import util
from BD.sqlite import BD
import json
import random

# lectura de datos EMPATIA
from Problem.EMPATIA.database.prepare_dataset import prepare_47vol_solap

# lectura modelo KNN 
from Problem.EMPATIA.model.ml_model import get_metrics_voluntaria
from Problem.EMPATIA.model.hyperparameter_optimization import load_parameters

def totalFeature():
    return 57

def factibilidad(individuo):
    suma = np.sum(individuo)
    if suma > 0:
        return True
    else:
        return False

def nuevaSolucion():
    return np.random.randint(low=0, high=2, size = totalFeature())

def get_fitness(loader, individuo, problema, opt, voluntaria):
    # Return the confusion matrix (TN, FP, FN, TP)   
    
    scores = get_metrics_voluntaria(loader, 
                        selected_features=individuo,
                        optimal_parameters=opt,
                        threshold = 0.126,
                        id = loader.dataset['vol_id'].unique()[voluntaria],
                        opt_params = voluntaria)
    
    # # tn = scores[0]
    # # fp = scores[1]
    # # fn = scores[2]
    # # tp = scores[3]
    
    # metricas merge
    tn = scores['tn_merge'][0]
    fp = scores['fp_merge'][0]
    fn = scores['fn_merge'][0]
    tp = scores['tp_merge'][0]    
    
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    errorRate   = 1 - accuracy
    mcc         = ( (tp * tn) - (fp * fn) ) / ( np.sqrt( (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) ) )

    precision   = tp / (tp + fp)
    recall      = tp / (tp + fn)
    f1          = 2 * ( (precision * recall) / (precision + recall) )
    
    fitness = 1 - f1
    
    if np.isnan(fitness):
        fitness = 10000
    if np.isnan(accuracy):
        accuracy = -1
    if np.isnan(errorRate):
        errorRate = 10000
    if np.isnan(mcc):
        mcc = 10000
    if np.isnan(precision):
        precision = -1
    if np.isnan(recall):
        recall = -1
    if np.isnan(f1):
        f1 = -1
    
    return np.round(fitness,3), np.round(accuracy,3), np.round(f1,3), np.round(precision,3), np.round(recall,3), np.round(mcc,3), np.round(errorRate,3), sum(individuo)
 

def solverEMPATIA_Voluntaria(id, mh, maxIter, pop, ds, instancia):
    data_dir = './Problem/EMPATIA/'
    
    opt = load_parameters(data_dir, "optimal_parameters_47sol")
    
    dirResult = './Resultados/'
    # tomo el tiempo inicial de la ejecucion
    initialTime = time.time()
    
    problema = instancia
    
    voluntaria = int(problema.split("-")[1].replace("V","")) - 1
    
    # LOAD DATA    
    # 47vol_solap : 47 voluntarias con solapamiento
    loader = prepare_47vol_solap(data_dir)
    
    tiempoInicializacion1 = time.time()

    print("------------------------------------------------------------------------------------------------------")
    print("RESOLVIENDO PROBLEMA EMPATIA-9 con voluntaria "+str(voluntaria))
    
    results = open(dirResult+mh+"_"+problema+"_"+str(id)+".csv", "w")
    results.write(
        f'iter,fitness,time,accuracy,f1-score,precision,recall,mcc,errorRate,TFS,XPL,XPT,DIV\n'
    )
    
    # Genero una población inicial binaria, esto ya que nuestro problema es binario
    # poblacion = np.random.randint(low=0, high=2, size = (pop,totalFeature()))
    poblacion = np.ones(shape = (pop,totalFeature()))
    
    # Genero una población inicial basada en el metodo de filtrado
    # poblacion = generarPoblacionInicialFilter()

    maxDiversidad = diversidadHussain(poblacion)
    XPL , XPT, state = porcentajesXLPXPT(maxDiversidad, maxDiversidad)

    # Genero un vector donde almacenaré los fitness de cada individuo
    fitness                 = np.zeros(pop)
    accuracy                = np.zeros(pop)
    f1Score                 = np.zeros(pop)
    presicion               = np.zeros(pop)
    recall                  = np.zeros(pop)
    mcc                     = np.zeros(pop)
    errorRate               = np.zeros(pop)
    totalFeatureSelected    = np.zeros(pop)
    

    # Genero un vetor dedonde tendré mis soluciones rankeadas
    solutionsRanking = np.zeros(pop)
    # calculo de factibilidad de cada individuo y calculo del fitness inicial
    for i in range(poblacion.__len__()):
        if not factibilidad(poblacion[i]): # sin caracteristicas seleccionadas
                # nueva solucion
                poblacion[i] = nuevaSolucion()
                
                # nueva solucion basada en el metodo de filtrado = 
                # poblacion[i] = nuevaSolucionFiltrado()
        fitness[i], accuracy[i], f1Score[i], presicion[i], recall[i], mcc[i], errorRate[i], totalFeatureSelected[i] = get_fitness(loader, poblacion[i], problema, opt, voluntaria)
        
    solutionsRanking = np.argsort(fitness) # rankings de los mejores fitnes
    bestIdx = solutionsRanking[0]
    # DETERMINO MI MEJOR SOLUCION Y LA GUARDO 
    Best = poblacion[bestIdx].copy()
    BestFitness = fitness[bestIdx]
    BestAccuracy = accuracy[bestIdx]
    BestF1Score = f1Score[bestIdx]
    BestPresicion = presicion[bestIdx]
    BestRecall = recall[bestIdx]
    BestMcc = mcc[bestIdx]
    bestErrorRate = errorRate[bestIdx]
    bestTFS = totalFeatureSelected[bestIdx]
    
    # PARA MFO
    BestFitnessArray = fitness[solutionsRanking] 
    accuracyArray                = np.zeros(pop)
    f1ScoreArray                 = np.zeros(pop)
    presicionArray               = np.zeros(pop)
    recallArray                  = np.zeros(pop)
    mccArray                     = np.zeros(pop)
    errorRateArray               = np.zeros(pop)
    totalFeatureSelectedArray    = np.zeros(pop)
    bestSolutions = poblacion[solutionsRanking]
    
    matrixBin = poblacion.copy()

    tiempoInicializacion2 = time.time()

    # mostramos nuestro fitness iniciales
    print("------------------------------------------------------------------------------------------------------")
    print("fitness iniciales: "+str(fitness))
    print("Best fitness inicial: "+str(np.min(fitness)))
    print("------------------------------------------------------------------------------------------------------")
    if mh == "GA":
        print("COMIENZA A TRABAJAR LA METAHEURISTICA "+mh)
    else:
        print("COMIENZA A TRABAJAR LA METAHEURISTICA "+mh+ " / Binarizacion: "+ str(ds))
    print("------------------------------------------------------------------------------------------------------")
    print(
        f'i: 0'+
        f', b: {str(BestFitness)}'+
        f', t: {str(round(tiempoInicializacion2-tiempoInicializacion1,3))}'+
        f', a: {str(BestAccuracy)}'+
        f', fs: {str(BestF1Score)}'+
        f', p: {str(BestPresicion)}'+
        f', rc: {str(BestRecall)}'+
        f', mcc: {str(BestMcc)}'+
        f', eR: {str(bestErrorRate)}'+
        f', TFS: {str(bestTFS)}'+
        f', XPL: {str(XPL)}'+
        f', XPT: {str(XPT)}'+
        f', DIV: {str(maxDiversidad)}'
    )
    results.write(
        f'0,{str(BestFitness)},{str(round(tiempoInicializacion2-tiempoInicializacion1,3))},{str(BestAccuracy)},{str(BestF1Score)},{str(BestPresicion)},{str(BestRecall)},{str(BestMcc)},{str(bestErrorRate)},{str(bestTFS)},{str(XPL)},{str(XPT)},{str(maxDiversidad)}\n'
    )
    
    # for de iteraciones
    for iter in range(0, maxIter):
        
        # obtengo mi tiempo inicial
        timerStart = time.time()
        
        if mh == "MFO":
            for i in range(bestSolutions.__len__()):
                BestFitnessArray[i], accuracyArray[i], f1ScoreArray[i], presicionArray[i], recallArray[i], mccArray[i], errorRateArray[i], totalFeatureSelectedArray[i] = get_fitness(loader, poblacion[i], problema, opt)
        
        
        # perturbo la poblacion con la metaheuristica, pueden usar SCA y GWO
        # en las funciones internas tenemos los otros dos for, for de individuos y for de dimensiones
        # print(poblacion)
        if mh == "SCA":
            poblacion = iterarSCA(maxIter, iter, totalFeature(), poblacion.tolist(), Best.tolist())
        if mh == "GWO":
            poblacion = iterarGWO(maxIter, iter, totalFeature(), poblacion.tolist(), fitness.tolist(), 'MIN')
        if mh == 'WOA':
            poblacion = iterarWOA(maxIter, iter, totalFeature(), poblacion.tolist(), Best.tolist())
        if mh == 'PSA':
            poblacion = iterarPSA(maxIter, iter, totalFeature(), poblacion.tolist(), Best.tolist())
        if mh == 'MFO':
            poblacion, bestSolutions = iterarMFO(maxIter, iter, totalFeature(), len(poblacion), poblacion, bestSolutions, fitness, BestFitnessArray)
        if mh == "GA":
            poblacion = iterarGA(poblacion.tolist(), fitness)
        
        # Binarizo, calculo de factibilidad de cada individuo y calculo del fitness
        for i in range(poblacion.__len__()):
            
            if mh != "GA":
                poblacion[i] = b.aplicarBinarizacion(poblacion[i].tolist(), ds[0], ds[1], Best, matrixBin[i].tolist())

            if not factibilidad(poblacion[i]): # sin caracteristicas seleccionadas
                # nueva solucion
                poblacion[i] = nuevaSolucion()
                
                # nueva solucion basada en el metodo de filtrado = 
                # poblacion[i] = nuevaSolucionFiltrado()
            
            fitness[i], accuracy[i], f1Score[i], presicion[i], recall[i], mcc[i], errorRate[i], totalFeatureSelected[i] = get_fitness(loader, poblacion[i], problema, opt, voluntaria)

        solutionsRanking = np.argsort(fitness) # rankings de los mejores fitness
        #Conservo el Best
        if fitness[solutionsRanking[0]] < BestFitness:
            bestIdx = solutionsRanking[0]
            BestFitness = fitness[solutionsRanking[0]]
            Best = poblacion[solutionsRanking[0]]
            BestAccuracy = accuracy[bestIdx]
            BestF1Score = f1Score[bestIdx]
            BestPresicion = presicion[bestIdx]
            BestRecall = recall[bestIdx]
            BestMcc = mcc[bestIdx]
            bestErrorRate = errorRate[bestIdx]
            bestTFS = totalFeatureSelected[bestIdx]
        matrixBin = poblacion.copy()

        div_t = diversidadHussain(poblacion)

        if maxDiversidad < div_t:
            maxDiversidad = div_t

        XPL , XPT, state = porcentajesXLPXPT(div_t, maxDiversidad)

        timerFinal = time.time()
        # calculo mi tiempo para la iteracion t
        timeEjecuted = timerFinal - timerStart

        print(
        f'i: {str(iter+1)}'+
        f', b: {str(BestFitness)}'+
        f', t: {str(round(timeEjecuted,3))}'+
        f', a: {str(BestAccuracy)}'+
        f', fs: {str(BestF1Score)}'+
        f', p: {str(BestPresicion)}'+
        f', rc: {str(BestRecall)}'+
        f', mcc: {str(BestMcc)}'+
        f', eR: {str(bestErrorRate)}'+
        f', TFS: {str(bestTFS)}'+
        f', XPL: {str(XPL)}'+
        f', XPT: {str(XPT)}'+
        f', DIV: {str(div_t)}'
        )
        results.write(
            f'{str(iter+1)},{str(BestFitness)},{str(round(timeEjecuted,3))},{str(BestAccuracy)},{str(BestF1Score)},{str(BestPresicion)},{str(BestRecall)},{str(BestMcc)},{str(bestErrorRate)},{str(bestTFS)},{str(XPL)},{str(XPT)},{str(div_t)}\n'
        )

    print("------------------------------------------------------------------------------------------------------")
    print("Best fitness: "+str(BestFitness))
    print("Cantidad de caracteristicas seleccionadas: "+str(bestTFS))
    print("------------------------------------------------------------------------------------------------------")
    finalTime = time.time()
    tiempoEjecucion = finalTime - initialTime
    print("Tiempo de ejecucion (s): "+str(tiempoEjecucion))
    # results.write("Tiempo de ejecucion (s): "+str(tiempoEjecucion))
    print("Solucion: "+str(Best.tolist()))
    results.close()
    
    binary = util.convert_into_binary(dirResult+mh+"_"+problema+"_"+str(id)+".csv")

    nombre_archivo = mh+"_"+problema

    bd = BD()
    bd.insertarIteraciones(nombre_archivo, binary, id)
    bd.insertarResultados(BestFitness, tiempoEjecucion, Best, id)
    bd.actualizarExperimento(id, 'terminado')
    
    os.remove(dirResult+mh+"_"+problema+"_"+str(id)+".csv")