import numpy as np
import os
from Problem.FS.Problem import FeatureSelection as fs
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

from MachineLearning.QLearning import QLearning




def solverFSML(id, mh, maxIter, pop, instancia, clasificador, parametrosC, paramsML, ml):
    
    dirResult = './Resultados/'
    instance = fs(instancia)

    # tomo el tiempo inicial de la ejecucion
    initialTime = time.time()
    
    tiempoInicializacion1 = time.time()

    print("------------------------------------------------------------------------------------------------------")
    print("instancia FS a resolver: "+instancia)


    results = open(dirResult+mh+"_"+instancia.split(".")[0]+"_"+str(id)+".csv", "w")
    results.write(
        f'iter,fitness,time,accuracy,f1-score,precision,recall,mcc,errorRate,TFS,XPL,XPT,DIV,state,action\n'
    )

    # Genero una población inicial binaria, esto ya que nuestro problema es binario
    poblacion = np.random.randint(low=0, high=2, size = (pop,instance.getTotalFeature()))

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

        if not instance.factibilidad(poblacion[i]): # sin caracteristicas seleccionadas
                poblacion[i] = instance.nuevaSolucion()

        seleccion = np.where(poblacion[i] == 1)[0]
        fitness[i], accuracy[i], f1Score[i], presicion[i], recall[i], mcc[i], errorRate[i], totalFeatureSelected[i] = instance.fitness(seleccion, clasificador, parametrosC)

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
    print("COMIENZA A TRABAJAR LA METAHEURISTICA "+mh+ " / Machine Learning include: "+ml)
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
        
        # Q-Learning - selección de una acción
        action = agente.getAction(state)
        ds = paramsML['DS_actions'][action].split("-")
        
        if mh == "MFO":
            for i in range(bestSolutions.__len__()):
                seleccion = np.where(bestSolutions[i] == 1)[0]
                BestFitnessArray[i], accuracyArray[i], f1ScoreArray[i], presicionArray[i], recallArray[i], mccArray[i], errorRateArray[i], totalFeatureSelectedArray[i] = instance.fitness(seleccion, clasificador, parametrosC)
        
        # perturbo la poblacion con la metaheuristica, pueden usar SCA y GWO
        # en las funciones internas tenemos los otros dos for, for de individuos y for de dimensiones
        # print(poblacion)
        if mh == "SCA":
            poblacion = iterarSCA(maxIter, iter, instance.getTotalFeature(), poblacion.tolist(), Best.tolist())
        if mh == "GWO":
            poblacion = iterarGWO(maxIter, iter, instance.getTotalFeature(), poblacion.tolist(), fitness.tolist(), 'MIN')
        if mh == 'WOA':
            poblacion = iterarWOA(maxIter, iter, instance.getTotalFeature(), poblacion.tolist(), Best.tolist())
        if mh == 'PSA':
            poblacion = iterarPSA(maxIter, iter, instance.getTotalFeature(), poblacion.tolist(), Best.tolist())
        if mh == "MFO":
            poblacion, bestSolutions = iterarMFO(maxIter, iter, instance.getTotalFeature(), len(poblacion), poblacion, bestSolutions, fitness, BestFitnessArray )
        
        # Binarizo, calculo de factibilidad de cada individuo y calculo del fitness
        for i in range(poblacion.__len__()):
            poblacion[i] = b.aplicarBinarizacion(poblacion[i].tolist(), ds[0], ds[1], Best, matrixBin[i].tolist())

            if not instance.factibilidad(poblacion[i]): # sin caracteristicas seleccionadas
                poblacion[i] = instance.nuevaSolucion()
            
            seleccion = np.where(poblacion[i] == 1)[0]
            fitness[i], accuracy[i], f1Score[i], presicion[i], recall[i], mcc[i], errorRate[i], totalFeatureSelected[i] = instance.fitness(seleccion, clasificador, parametrosC)

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
    print("Cantidad de acciones selecionadas: "+str(agente.getVisitas()))
    print("------------------------------------------------------------------------------------------------------")
    finalTime = time.time()
    tiempoEjecucion = finalTime - initialTime
    print("Tiempo de ejecucion (s): "+str(tiempoEjecucion))
    # results.write("Tiempo de ejecucion (s): "+str(tiempoEjecucion))
    print("Solucion: "+str(Best.tolist()))
    results.close()

    binary = util.convert_into_binary(dirResult+mh+"_"+instancia.split(".")[0]+"_"+str(id)+".csv")

    nombre_archivo = mh+"_"+instancia.split(".")[0]+"_"+ml

    bd = BD()
    bd.insertarIteraciones(nombre_archivo, binary, id)
    bd.insertarResultados(BestFitness, tiempoEjecucion, Best, id)
    bd.actualizarExperimento(id, 'terminado')
    
    os.remove(dirResult+mh+"_"+instancia.split(".")[0]+"_"+str(id)+".csv")

