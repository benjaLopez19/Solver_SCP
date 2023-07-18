import numpy as np
import os
from Metaheuristics.GWO import iterarGWO 
from Metaheuristics.SCA import iterarSCA
from Metaheuristics.WOA import iterarWOA
from Metaheuristics.PSA import iterarPSA
from Metaheuristics.MFO import iterarMFO
from Diversity.hussainDiversity import diversidadHussain
from Diversity.XPLXTP import porcentajesXLPXPT
import time
from Discretization import discretization as b
from util import util
from BD.sqlite import BD
import math

# lectura de datos EMPATIA
from Problem.EMPATIA.database.emotional_dataset import EmotionalDataset
from Problem.EMPATIA.database.basic_loader import BasicLoader

# lectura modelo KNN 
from Problem.EMPATIA.model.ml_model import get_metrics
from Problem.EMPATIA.model.hyperparameter_optimization import load_parameters

# Q-Learning
from MachineLearning.QLearning import QLearning

def totalFeature():
    return 57

def factibilidad(individuo):
    suma = np.sum(individuo)
    if suma > 0:
        return True
    else:
        return False

def get_fitness(loader, individuo, problema, opt):
    # Return the confusion matrix (TN, FP, FN, TP)
    
    if problema == 'EMPATIA' or problema == 'EMPATIA-2':
        scores = get_metrics(loader, 
                            neighbors = 20,
                            missclassification_cost = 1.6,
                            selected_features=individuo)
    if problema == 'EMPATIA-3' or problema == 'EMPATIA-4':
        scores = get_metrics(loader, 
                            missclassification_cost = 1.6,
                            selected_features=individuo,
                            optimal_parameters=opt)
    
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
    
    if problema == 'EMPATIA' or problema == 'EMPATIA-3':
        fitness = (alpha_1 * errorRate) + (alpha_2 * (sum(individuo)/totalFeature()))
    if problema == 'EMPATIA-2' or problema == 'EMPATIA-4':
        fitness = 1 - f1
    
    return np.round(fitness,3), np.round(accuracy,3), np.round(f1,3), np.round(precision,3), np.round(recall,3), np.round(mcc,3), np.round(errorRate,3), sum(individuo)
    
def nuevaSolucion():
    return np.random.randint(low=0, high=2, size = totalFeature())

def solverEMPATIAML(id, mh, maxIter, pop, instancia, paramsML, ml):
    
    opt = load_parameters("./Problem/EMPATIA/model/")
    
    dirResult = './Resultados/'
    # tomo el tiempo inicial de la ejecucion
    initialTime = time.time()
    
    problema = instancia
    
    # LOAD DATA
    data_dir = './Problem/EMPATIA/'
    dataset = EmotionalDataset(data_dir + 'features_47Vol_CSVs',
                                data_dir + 'labels')
    REMOVE_TEST = [4, 5, 14, 32, 39]
    set_vol = dataset.df_data.index.get_level_values('vol_id').unique()
    non_valid_volunteers = [set_vol[i] for i in REMOVE_TEST];
    dataset.filter(filter_idx = non_valid_volunteers)
    loader = BasicLoader(dataset, norm = True)
    loader.dataset.reset_index(drop = True, inplace = True)
    
    
    tiempoInicializacion1 = time.time()

    print("------------------------------------------------------------------------------------------------------")
    print("RESOLVIENDO PROBLEMA "+problema)


    results = open(dirResult+mh+"_"+problema+"_"+str(id)+".csv", "w")
    results.write(
        f'iter,fitness,time,accuracy,f1-score,precision,recall,mcc,errorRate,TFS,XPL,XPT,DIV,state,action\n'
    )
    
    # Genero una población inicial binaria, esto ya que nuestro problema es binario
    poblacion = np.random.randint(low=0, high=2, size = (pop,totalFeature()))
    # poblacion = np.ones(shape=(pop,totalFeature()))

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
                poblacion[i] = nuevaSolucion()
        fitness[i], accuracy[i], f1Score[i], presicion[i], recall[i], mcc[i], errorRate[i], totalFeatureSelected[i] = get_fitness(loader, poblacion[i], problema, opt)
    
    
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
    
    # Q-Learning - inicialización
    agente = QLearning(maxIter, paramsML)
    
    matrixBin = poblacion.copy()

    tiempoInicializacion2 = time.time()

    # mostramos nuestro fitness iniciales
    print("------------------------------------------------------------------------------------------------------")
    print("fitness iniciales: "+str(fitness))
    print("Best fitness inicial: "+str(np.min(fitness)))
    print("------------------------------------------------------------------------------------------------------")
    print("COMIENZA A TRABAJAR LA METAHEURISTICA "+mh+ " / Machine Learning include: "+ ml)
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
        f', DIV: {str(maxDiversidad)}'+
        f', state: {str(0)}'+
        f', action: {str(-1)}'
    )
    results.write(
        f'0,{str(BestFitness)},{str(round(tiempoInicializacion2-tiempoInicializacion1,3))},{str(BestAccuracy)},{str(BestF1Score)},{str(BestPresicion)},{str(BestRecall)},{str(BestMcc)},{str(bestErrorRate)},{str(bestTFS)},{str(XPL)},{str(XPT)},{str(maxDiversidad)},{str(0)},{str(-1)}\n'
    )
    
    # for de iteraciones
    for iter in range(0, maxIter):
        
        # obtengo mi tiempo inicial
        timerStart = time.time()
        
        # Q-Learning - seleccion de una acción
        action = agente.getAction(state)
        ds = paramsML['DS_actions'][action].split("-")
        
        
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
            
            
        # Binarizo, calculo de factibilidad de cada individuo y calculo del fitness
        for i in range(poblacion.__len__()):
            poblacion[i] = b.aplicarBinarizacion(poblacion[i].tolist(), ds[0], ds[1], Best, matrixBin[i].tolist())

            if not factibilidad(poblacion[i]): # sin caracteristicas seleccionadas
                poblacion[i] = nuevaSolucion()
            
            fitness[i], accuracy[i], f1Score[i], presicion[i], recall[i], mcc[i], errorRate[i], totalFeatureSelected[i] = get_fitness(loader, poblacion[i], problema, opt)

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

        # Q-Learning - actualización Q-Table
        agente.updateQtable(np.min(fitness), action, state, iter)

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
        f', DIV: {str(div_t)}'+
        f', state: {str(state)}'+
        f', action: {str(action)}'
        )
        results.write(
            f'{str(iter+1)},{str(BestFitness)},{str(round(timeEjecuted,3))},{str(BestAccuracy)},{str(BestF1Score)},{str(BestPresicion)},{str(BestRecall)},{str(BestMcc)},{str(bestErrorRate)},{str(bestTFS)},{str(XPL)},{str(XPT)},{str(div_t)},{str(state)},{str(action)}\n'
        )

    print("------------------------------------------------------------------------------------------------------")
    print("Best fitness: "+str(BestFitness))
    print("Cantidad de caracteristicas seleccionadas: "+str(bestTFS))
    print("Cantidad de acciones seleccionadas: "+str(agente.getVisitas()))
    print("------------------------------------------------------------------------------------------------------")
    finalTime = time.time()
    tiempoEjecucion = finalTime - initialTime
    print("Tiempo de ejecucion (s): "+str(tiempoEjecucion))
    # results.write("Tiempo de ejecucion (s): "+str(tiempoEjecucion))
    print("Solucion: "+str(Best.tolist()))
    results.close()
    
    binary = util.convert_into_binary(dirResult+mh+"_"+problema+"_"+str(id)+".csv")

    nombre_archivo = mh+"_"+problema+"_"+ml

    bd = BD()
    bd.insertarIteraciones(nombre_archivo, binary, id)
    bd.insertarResultados(BestFitness, tiempoEjecucion, Best, id)
    bd.actualizarExperimento(id, 'terminado')
    
    os.remove(dirResult+mh+"_"+problema+"_"+str(id)+".csv")