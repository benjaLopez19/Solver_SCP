from Problem.EMPATIA.model.fs_simple_methods import (create_filter_list, create_LASSO_foreach,
                                                    create_LASSO_list, create_wrapper_list)
import numpy as np
import time
# # lectura de datos EMPATIA
from Problem.EMPATIA.database.prepare_dataset import prepare_47vol_solap,  prepare_47vol_nosolap,  prepare_100vol_solap,  prepare_100vol_nosolap

# lectura modelo KNN 
from Problem.EMPATIA.model.ml_model import get_metrics
from Problem.EMPATIA.model.hyperparameter_optimization import load_parameters

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

def wrapper(instancia, id):

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

    performance  = open("Sequential_feature_selector/sequential_feature_selector_performance_merge_"+problema+"_"+str(id)+".csv", "w")
    solutions    = open("Sequential_feature_selector/sequential_feature_selector_solution_merge_"+problema+"_"+str(id)+".csv", "w")
    performance.write(f'''tfs,time,accuracy,f-score,presicion,recall,errorRate,mcc\n''')
    solutions.write("solution;fitness\n")

    features = 1

    while features < totalFeature():
        

        timeInicial = time .time()   

        seleccion = create_wrapper_list(loader = loader, n_variables = features, label_to_predict = 'y')

        fitness, accuracy, f1Score, presicion, recall, mcc, errorRate, totalFeatureSelected = get_fitness(loader, seleccion, problema, opt)

        timeFinal = time.time()

        performance.write(f'''{totalFeatureSelected},{np.around(timeFinal-timeInicial,3)},{accuracy},{f1Score},{presicion},{recall},{errorRate},{mcc}\n''')
        solutions.write(str(seleccion)+";"+str(fitness)+"\n")
        
        print(f'''{problema},{totalFeatureSelected},{np.around(timeFinal-timeInicial,3)},{accuracy},{f1Score},{presicion},{recall},{errorRate},{mcc}''')
        
        features+=1


    performance.close()
    solutions.close()


ids = [3,7,11,15,19,23,27,31]

for i in ids:
    wrapper('EMPATIA-9', i)  