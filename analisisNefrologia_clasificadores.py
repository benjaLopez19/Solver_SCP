import numpy as np
import pandas as pd
from util.util import selectionSort
from BD.sqlite import BD
from Problem.FS.Problem import FeatureSelection as fs

bd = BD()


instancia = 'dat_3_3_1'
clasificador = 'KNN'

instance = fs(instancia)

binarizaciones = ["100","200","300","400","500"]

for bss in binarizaciones:

    blob = bd.obtenerMejoresArchivosconClasificadorBSS(instancia,"",clasificador,bss)
    
    for d in blob:
        
        # print(d[9].replace('[','').replace(']','').split(','))
        # lista = list(d[9])
        # print(lista)
        
        solution = np.array(d[9].replace('[','').replace(']','').replace(' ','').split(','))
        
        # print(solution)
        
        j = 0
        for i in solution:
            solution[j] = int(float(i))
            j+=1
        
        solution = np.array(solution, dtype=np.int16)
        
        # print(solution)
        
        seleccion = np.where(solution == 1)[0]
        
        # print(seleccion)
        fitness, accuracy, f1Score, presicion, recall, mcc, errorRate, totalFeatureSelected = instance.fitness(seleccion, 'RandomForest', 'k:5')
        print(f'''f1Score RandomForest: {f1Score}''')
        
        fitness, accuracy, f1Score, presicion, recall, mcc, errorRate, totalFeatureSelected = instance.fitness(seleccion, 'Xgboost', 'k:5')
        print(f'''f1Score Xgboost: {f1Score}''')