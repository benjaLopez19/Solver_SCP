import numpy as np
import os
import time
from Problem.MKP.problem import mkp
from Metaheuristics.GWO import iterarGWO
from Metaheuristics.PSA import iterarPSA
from Metaheuristics.SCA import iterarSCA
from Metaheuristics.WOA import iterarWOA
from Metaheuristics.GA import iterarGA
from Diversity.hussainDiversity import diversidadHussain
from Diversity.XPLXTP import porcentajesXLPXPT
from Discretization import discretization as b
from util import util 
from BD.sqlite import BD

def solverMKP(id, mh, maxIter, pop, instancia, DS):
    
    dirResult = './Resultados/'
    
    instance = None
    
    if instancia.split("_")[0] == "mknap1":
        instance = mkp(instancia,1)
    if instancia.split("_")[0] == "mknap2":
        instance = mkp(instancia,2)
    
    # tomo el tiempo inicial de la ejecución
    initialTime = time.time()
    
    tiempoInicializacion1 = time.time()
    
    print("------------------------------------------------------------------------------------------------------")
    print("instancia MKP a resolver: "+instancia)
    
    results = open(dirResult+mh+"_"+instancia.split(".")[0]+"_"+str(id)+".csv", "w")
    results.write(
        f'iter,optimo,fitness,time,XPL,XPT,DIV\n'
    )
    
    # Genero una población inicial binaria, esto ya que nuestro problema es binario
    poblacion = np.random.randint(low=0, high=2, size = (pop, instance.getElements()))
    # poblacion = np.zeros(shape = (pop, instance.getElements()))

    maxDiversidad = diversidadHussain(poblacion)
    XPL , XPT, state = porcentajesXLPXPT(maxDiversidad, maxDiversidad)
    
    # Genero un vector donde almacenaré los fitness de cada individuo
    fitness = np.zeros(pop)

    # Genero un vetor dedonde tendré mis soluciones rankeadas
    solutionsRanking = np.zeros(pop)
    
    # calculo de factibilidad de cada individuo y calculo del fitness inicial
    for i in range(poblacion.__len__()):
        flag = instance.test_factibilidad(poblacion[i])
        if not flag: #solucion infactible
            poblacion[i] = instance.repairSolution(poblacion[i])
            

        fitness[i] = instance.calcule_fitness(poblacion[i])
    
    aux = np.argsort(fitness)
    solutionsRanking = util.invertirArray(aux)
    bestRowAux = solutionsRanking[0]
    
    Best = poblacion[bestRowAux].copy()
    BestFitness = fitness[bestRowAux]
    
    matrixBin = poblacion.copy()
    
    tiempoInicializacion2 = time.time()
    
    # mostramos nuestro fitness iniciales
    print("------------------------------------------------------------------------------------------------------")
    print("fitness incial: "+str(fitness))
    print("Best fitness inicial: "+str(BestFitness))
    print("------------------------------------------------------------------------------------------------------")
    if mh == "GA":
        print("COMIENZA A TRABAJAR LA METAHEURISTICA "+mh)
    else :
        print("COMIENZA A TRABAJAR LA METAHEURISTICA "+mh+ " / Binarizacion: "+ str(DS))
    print("------------------------------------------------------------------------------------------------------")
    print("iteracion: "+
            str(0)+
            ", best: "+str(BestFitness)+
            ", optimo: "+str(instance.getOptimo())+
            ", time (s): "+str(round(tiempoInicializacion2-tiempoInicializacion1,3))+
            ", XPT: "+str(XPT)+
            ", XPL: "+str(XPL)+
            ", DIV: "+str(maxDiversidad))
    results.write(
        f'0,{instance.getOptimo()},{str(BestFitness)},{str(round(tiempoInicializacion2-tiempoInicializacion1,3))},{str(XPL)},{str(XPT)},{maxDiversidad}\n'
    )
    
    for iter in range(0, maxIter):
        # obtengo mi tiempo inicial
        timerStart = time.time()
        
        # perturbo la poblacion con la metaheuristica, pueden usar SCA y GWO
        # en las funciones internas tenemos los otros dos for, for de individuos y for de dimensiones
        
        if mh == "SCA":
            poblacion = iterarSCA(maxIter, iter, instance.getElements(), poblacion.tolist(), Best.tolist())
        if mh == "GWO":
            poblacion = iterarGWO(maxIter, iter, instance.getElements(), poblacion.tolist(), fitness.tolist(), 'MAX')
        if mh == 'WOA':
            poblacion = iterarWOA(maxIter, iter, instance.getElements(), poblacion.tolist(), Best.tolist())
        if mh == 'PSA':
            poblacion = iterarPSA(maxIter, iter, instance.getElements(), poblacion.tolist(), Best.tolist())
        if mh == "GA":
            poblacion = iterarGA(poblacion.tolist())
            
        # Binarizo, calculo de factibilidad de cada individuo y calculo del fitness
        for i in range(poblacion.__len__()):

            if mh != "GA":
                poblacion[i] = b.aplicarBinarizacion(poblacion[i].tolist(), DS[0], DS[1], Best, matrixBin[i].tolist())

            flag = instance.test_factibilidad(poblacion[i])
            # print(aux)
            if not flag: #solucion infactible
                poblacion[i] = instance.repairSolution(poblacion[i])
                

            fitness[i] = instance.calcule_fitness(poblacion[i])
        
        aux = np.argsort(fitness)
        solutionsRanking = util.invertirArray(aux)
        
        # Conservo el Best
        if fitness[solutionsRanking[0]] > BestFitness:
            BestFitness = fitness[solutionsRanking[0]]
            Best = poblacion[solutionsRanking[0]]
        matrixBin.copy()
        
        div_t = diversidadHussain(poblacion)
        
        if maxDiversidad < div_t:
            maxDiversidad = div_t
        
        XPL , XPT, state = porcentajesXLPXPT(div_t, maxDiversidad)

        timerFinal = time.time()
        # calculo mi tiempo para la iteracion t
        timeEjecuted = timerFinal - timerStart
        
        print("iteracion: "+
            str(iter+1)+
            ", best: "+str(BestFitness)+
            ", optimo: "+str(instance.getOptimo())+
            ", time (s): "+str(round(tiempoInicializacion2-tiempoInicializacion1,3))+
            ", XPT: "+str(XPT)+
            ", XPL: "+str(XPL)+
            ", DIV: "+str(div_t))
        results.write(
            f'{iter+1},{instance.getOptimo()},{str(BestFitness)},{str(round(tiempoInicializacion2-tiempoInicializacion1,3))},{str(XPL)},{str(XPT)},{div_t}\n'
        )
    print("------------------------------------------------------------------------------------------------------")
    print("Best fitness: "+str(BestFitness))
    print("Cantidad de elementos seleccionados: "+str(sum(Best)))
    print("------------------------------------------------------------------------------------------------------")
    finalTime = time.time()
    tiempoEjecucion = finalTime - initialTime
    print("Tiempo de ejecucion (s): "+str(tiempoEjecucion))
    results.close()
    
    binary = util.convert_into_binary(dirResult+mh+"_"+instancia.split(".")[0]+"_"+str(id)+".csv")

    nombre_archivo = mh+"_"+instancia.split(".")[0]

    bd = BD()
    bd.insertarIteraciones(nombre_archivo, binary, id)
    bd.insertarResultados(BestFitness, tiempoEjecucion, Best, id)
    bd.actualizarExperimento(id, 'terminado')
    
    os.remove(dirResult+mh+"_"+instancia.split(".")[0]+"_"+str(id)+".csv")