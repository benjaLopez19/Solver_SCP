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
from Problem.EMPATIA.database.prepare_dataset import prepare_47vol_solap,  prepare_47vol_nosolap,  prepare_100vol_solap,  prepare_100vol_nosolap

# lectura modelo KNN 
from Problem.EMPATIA.model.ml_model import get_metrics
from Problem.EMPATIA.model.hyperparameter_optimization import load_parameters


def leerJsonFilter():

    features    = []
    f_score     = []
    accuracy    = []
    mcc         = []
    error_rate  = []
    precision   = []
    recall      = []

    with open('Problem/EMPATIA/filters/filters.json') as file:
        data = json.load(file)
        
        for feature in data['filter']:
            features.append(np.array(feature['features']))
            f_score.append(feature['f-score'])
            accuracy.append(feature['accuracy'])
            mcc.append(feature['mcc'])
            error_rate.append(feature['error rate'])
            precision.append(feature['precision'])
            recall.append(feature['recall'])
            
    return features, f_score, accuracy, mcc, error_rate, precision, recall
    
def generarPoblacionInicialFilter():

    features, f_score, accuracy, mcc, error_rate, precision, recall = leerJsonFilter()

    mod_features    = []
    mod_f_score     = []
    # mod_accuracy    = []
    # mod_mcc         = []
    # mod_error_rate  = []
    # mod_precision   = []
    # mod_recall      = []

    for i in range(len(features)):
        existe = False
        for j in range(len(mod_features)):
            if np.array_equal(mod_features[j], features[i]):
                existe = True
                
        if not existe:
            mod_features.append(features[i])
            mod_f_score.append(f_score[i])
            # mod_accuracy.append(accuracy[i])
            # mod_mcc.append(mcc[i])
            # mod_error_rate.append(error_rate[i])
            # mod_precision.append(precision[i])
            # mod_recall.append(recall[i])
            
    ordenados = np.argsort(mod_f_score)
    limit = 10
    total = len(ordenados)
    poblacion = []
    for i in reversed(ordenados[total-limit:total]):
        poblacion.append(mod_features[i])
    

    return np.array(poblacion)

def nuevaSolucionFiltrado():

    features, f_score, accuracy, mcc, error_rate, precision, recall = leerJsonFilter()

    mod_features    = []
    mod_f_score     = []
    # mod_accuracy    = []
    # mod_mcc         = []
    # mod_error_rate  = []
    # mod_precision   = []
    # mod_recall      = []

    for i in range(len(features)):
        existe = False
        for j in range(len(mod_features)):
            if np.array_equal(mod_features[j], features[i]):
                existe = True
                
        if not existe:
            mod_features.append(features[i])
            mod_f_score.append(f_score[i])
            # mod_accuracy.append(accuracy[i])
            # mod_mcc.append(mcc[i])
            # mod_error_rate.append(error_rate[i])
            # mod_precision.append(precision[i])
            # mod_recall.append(recall[i])
    return random.choice(np.array(mod_features))

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
    
    # tn = scores[0]
    # fp = scores[1]
    # fn = scores[2]
    # tp = scores[3]
    
    # metricas merge
    tn = scores[4]
    fp = scores[5]
    fn = scores[6]
    tp = scores[7]
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
    
def nuevaSolucion():
    return np.random.randint(low=0, high=2, size = totalFeature())

def solverEMPATIA(ids, mh, maxIter, pop, ds, instancia):
    
    # optimal_parameters_47sol
    # optimal_parameters_47nosol
    # optimal_parameters_100sol
    # optimal_parameters_100nosol
    
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
    
    dirResult = './Resultados/'
    # tomo el tiempo inicial de la ejecucion
    initialTime = time.time()
    
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
    
    tiempoInicializacion1 = time.time()

    print("------------------------------------------------------------------------------------------------------")
    print("RESOLVIENDO PROBLEMA "+problema)


    for id in ids:
        results = open(dirResult+mh+"_"+problema+"_"+str(id)+".csv", "w")
        results.write(
            f'iter,fitness,time,accuracy,f1-score,precision,recall,mcc,errorRate,TFS,XPL,XPT,DIV\n'
        )
        results.close()
    
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
    for id in ids:
        results = open(dirResult+mh+"_"+problema+"_"+str(id)+".csv", "a")
        results.write(
            f'0,{str(BestFitness)},{str(round(tiempoInicializacion2-tiempoInicializacion1,3))},{str(BestAccuracy)},{str(BestF1Score)},{str(BestPresicion)},{str(BestRecall)},{str(BestMcc)},{str(bestErrorRate)},{str(bestTFS)},{str(XPL)},{str(XPT)},{str(maxDiversidad)}\n'
        )
        results.close()
    
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
        
        if iter+1 < 100:
            results = open(dirResult+mh+"_"+problema+"_"+str(ids[0])+".csv", "a")
            results.write(
                f'{str(iter+1)},{str(BestFitness)},{str(round(timeEjecuted,3))},{str(BestAccuracy)},{str(BestF1Score)},{str(BestPresicion)},{str(BestRecall)},{str(BestMcc)},{str(bestErrorRate)},{str(bestTFS)},{str(XPL)},{str(XPT)},{str(div_t)}\n'
            )
            results.close()
        
        if iter+1 < 200:
            results = open(dirResult+mh+"_"+problema+"_"+str(ids[1])+".csv", "a")
            results.write(
                f'{str(iter+1)},{str(BestFitness)},{str(round(timeEjecuted,3))},{str(BestAccuracy)},{str(BestF1Score)},{str(BestPresicion)},{str(BestRecall)},{str(BestMcc)},{str(bestErrorRate)},{str(bestTFS)},{str(XPL)},{str(XPT)},{str(div_t)}\n'
            )
            results.close()
            
        if iter+1 < 300:
            results = open(dirResult+mh+"_"+problema+"_"+str(ids[2])+".csv", "a")
            results.write(
                f'{str(iter+1)},{str(BestFitness)},{str(round(timeEjecuted,3))},{str(BestAccuracy)},{str(BestF1Score)},{str(BestPresicion)},{str(BestRecall)},{str(BestMcc)},{str(bestErrorRate)},{str(bestTFS)},{str(XPL)},{str(XPT)},{str(div_t)}\n'
            )
            results.close()
            
        if iter+1 < 400:
            results = open(dirResult+mh+"_"+problema+"_"+str(ids[3])+".csv", "a")
            results.write(
                f'{str(iter+1)},{str(BestFitness)},{str(round(timeEjecuted,3))},{str(BestAccuracy)},{str(BestF1Score)},{str(BestPresicion)},{str(BestRecall)},{str(BestMcc)},{str(bestErrorRate)},{str(bestTFS)},{str(XPL)},{str(XPT)},{str(div_t)}\n'
            )
            results.close()
        
        if iter+1 < 500:
            results = open(dirResult+mh+"_"+problema+"_"+str(ids[4])+".csv", "a")
            results.write(
                f'{str(iter+1)},{str(BestFitness)},{str(round(timeEjecuted,3))},{str(BestAccuracy)},{str(BestF1Score)},{str(BestPresicion)},{str(BestRecall)},{str(BestMcc)},{str(bestErrorRate)},{str(bestTFS)},{str(XPL)},{str(XPT)},{str(div_t)}\n'
            )
            results.close()
            
        if iter+1 < 600:
            results = open(dirResult+mh+"_"+problema+"_"+str(ids[5])+".csv", "a")
            results.write(
                f'{str(iter+1)},{str(BestFitness)},{str(round(timeEjecuted,3))},{str(BestAccuracy)},{str(BestF1Score)},{str(BestPresicion)},{str(BestRecall)},{str(BestMcc)},{str(bestErrorRate)},{str(bestTFS)},{str(XPL)},{str(XPT)},{str(div_t)}\n'
            )
            results.close()

        if iter+1 < 700:
            results = open(dirResult+mh+"_"+problema+"_"+str(ids[6])+".csv", "a")
            results.write(
                f'{str(iter+1)},{str(BestFitness)},{str(round(timeEjecuted,3))},{str(BestAccuracy)},{str(BestF1Score)},{str(BestPresicion)},{str(BestRecall)},{str(BestMcc)},{str(bestErrorRate)},{str(bestTFS)},{str(XPL)},{str(XPT)},{str(div_t)}\n'
            )
            results.close()
            
        if iter+1 < 800:
            results = open(dirResult+mh+"_"+problema+"_"+str(ids[7])+".csv", "a")
            results.write(
                f'{str(iter+1)},{str(BestFitness)},{str(round(timeEjecuted,3))},{str(BestAccuracy)},{str(BestF1Score)},{str(BestPresicion)},{str(BestRecall)},{str(BestMcc)},{str(bestErrorRate)},{str(bestTFS)},{str(XPL)},{str(XPT)},{str(div_t)}\n'
            )
            results.close()
            
        if iter+1 < 900:
            results = open(dirResult+mh+"_"+problema+"_"+str(ids[8])+".csv", "a")
            results.write(
                f'{str(iter+1)},{str(BestFitness)},{str(round(timeEjecuted,3))},{str(BestAccuracy)},{str(BestF1Score)},{str(BestPresicion)},{str(BestRecall)},{str(BestMcc)},{str(bestErrorRate)},{str(bestTFS)},{str(XPL)},{str(XPT)},{str(div_t)}\n'
            )
            results.close()
            
        if iter+1 < 1000:
            results = open(dirResult+mh+"_"+problema+"_"+str(ids[9])+".csv", "a")
            results.write(
                f'{str(iter+1)},{str(BestFitness)},{str(round(timeEjecuted,3))},{str(BestAccuracy)},{str(BestF1Score)},{str(BestPresicion)},{str(BestRecall)},{str(BestMcc)},{str(bestErrorRate)},{str(bestTFS)},{str(XPL)},{str(XPT)},{str(div_t)}\n'
            )
            results.close()
            
            
        if iter+1 == 100:
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
            
            binary = util.convert_into_binary(dirResult+mh+"_"+problema+"_"+str(ids[0])+".csv")

            nombre_archivo = mh+"_"+problema

            bd = BD()
            bd.insertarIteraciones(nombre_archivo, binary, ids[0])
            bd.insertarResultados(BestFitness, tiempoEjecucion, Best, ids[0])
            bd.actualizarExperimento(ids[0], 'terminado')
            
            os.remove(dirResult+mh+"_"+problema+"_"+str(ids[0])+".csv")
        
        if iter+1 == 200:
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
            
            binary = util.convert_into_binary(dirResult+mh+"_"+problema+"_"+str(ids[1])+".csv")

            nombre_archivo = mh+"_"+problema

            bd = BD()
            bd.insertarIteraciones(nombre_archivo, binary, ids[1])
            bd.insertarResultados(BestFitness, tiempoEjecucion, Best, ids[1])
            bd.actualizarExperimento(ids[1], 'terminado')
            
            os.remove(dirResult+mh+"_"+problema+"_"+str(ids[1])+".csv")
            
        if iter+1 == 300:
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
            
            binary = util.convert_into_binary(dirResult+mh+"_"+problema+"_"+str(ids[2])+".csv")

            nombre_archivo = mh+"_"+problema

            bd = BD()
            bd.insertarIteraciones(nombre_archivo, binary, ids[2])
            bd.insertarResultados(BestFitness, tiempoEjecucion, Best, ids[2])
            bd.actualizarExperimento(ids[2], 'terminado')
            
            os.remove(dirResult+mh+"_"+problema+"_"+str(ids[2])+".csv")
            
        if iter+1 == 400:
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
            
            binary = util.convert_into_binary(dirResult+mh+"_"+problema+"_"+str(ids[3])+".csv")

            nombre_archivo = mh+"_"+problema

            bd = BD()
            bd.insertarIteraciones(nombre_archivo, binary, ids[3])
            bd.insertarResultados(BestFitness, tiempoEjecucion, Best, ids[3])
            bd.actualizarExperimento(ids[3], 'terminado')
            
            os.remove(dirResult+mh+"_"+problema+"_"+str(ids[3])+".csv")
            
        if iter+1 == 500:
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
            
            binary = util.convert_into_binary(dirResult+mh+"_"+problema+"_"+str(ids[4])+".csv")

            nombre_archivo = mh+"_"+problema

            bd = BD()
            bd.insertarIteraciones(nombre_archivo, binary, ids[4])
            bd.insertarResultados(BestFitness, tiempoEjecucion, Best, ids[4])
            bd.actualizarExperimento(ids[4], 'terminado')
            
            os.remove(dirResult+mh+"_"+problema+"_"+str(ids[4])+".csv")
            
        if iter+1 == 600:
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
            
            binary = util.convert_into_binary(dirResult+mh+"_"+problema+"_"+str(ids[5])+".csv")

            nombre_archivo = mh+"_"+problema

            bd = BD()
            bd.insertarIteraciones(nombre_archivo, binary, ids[5])
            bd.insertarResultados(BestFitness, tiempoEjecucion, Best, ids[5])
            bd.actualizarExperimento(ids[5], 'terminado')
            
            os.remove(dirResult+mh+"_"+problema+"_"+str(ids[5])+".csv")
            
        if iter+1 == 700:
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
            
            binary = util.convert_into_binary(dirResult+mh+"_"+problema+"_"+str(ids[6])+".csv")

            nombre_archivo = mh+"_"+problema

            bd = BD()
            bd.insertarIteraciones(nombre_archivo, binary, ids[6])
            bd.insertarResultados(BestFitness, tiempoEjecucion, Best, ids[6])
            bd.actualizarExperimento(ids[6], 'terminado')
            
            os.remove(dirResult+mh+"_"+problema+"_"+str(ids[6])+".csv")
            
        if iter+1 == 800:
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
            
            binary = util.convert_into_binary(dirResult+mh+"_"+problema+"_"+str(ids[7])+".csv")

            nombre_archivo = mh+"_"+problema

            bd = BD()
            bd.insertarIteraciones(nombre_archivo, binary, ids[7])
            bd.insertarResultados(BestFitness, tiempoEjecucion, Best, ids[7])
            bd.actualizarExperimento(ids[7], 'terminado')
            
            os.remove(dirResult+mh+"_"+problema+"_"+str(ids[7])+".csv")
            
        if iter+1 == 900:
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
            
            binary = util.convert_into_binary(dirResult+mh+"_"+problema+"_"+str(ids[8])+".csv")

            nombre_archivo = mh+"_"+problema

            bd = BD()
            bd.insertarIteraciones(nombre_archivo, binary, ids[8])
            bd.insertarResultados(BestFitness, tiempoEjecucion, Best, ids[8])
            bd.actualizarExperimento(ids[8], 'terminado')
            
            os.remove(dirResult+mh+"_"+problema+"_"+str(ids[8])+".csv")
        
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
    
    binary = util.convert_into_binary(dirResult+mh+"_"+problema+"_"+str(ids[9])+".csv")

    nombre_archivo = mh+"_"+problema

    bd = BD()
    bd.insertarIteraciones(nombre_archivo, binary, ids[9])
    bd.insertarResultados(BestFitness, tiempoEjecucion, Best, ids[9])
    bd.actualizarExperimento(ids[9], 'terminado')
    
    os.remove(dirResult+mh+"_"+problema+"_"+str(ids[9])+".csv")