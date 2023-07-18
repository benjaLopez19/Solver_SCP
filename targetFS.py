from Problem.FS.Problem import FeatureSelection

# lectura de datos EMPATIA
from Problem.EMPATIA.database.emotional_dataset import EmotionalDataset
from Problem.EMPATIA.database.basic_loader import BasicLoader

# lectura de modelo KNN
from Problem.EMPATIA.model.ml_model import get_metrics
from Problem.EMPATIA.model.hyperparameter_optimization import load_parameters

import numpy as np

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
        fitness = (alpha_1 * errorRate) + (alpha_2 * (sum(individuo)/57))
    if problema == 'EMPATIA-2' or problema == 'EMPATIA-4':
        fitness = 1 - f1
    
    return np.round(fitness,3), np.round(accuracy,3), np.round(f1,3), np.round(precision,3), np.round(recall,3), np.round(mcc,3), np.round(errorRate,3), sum(individuo)

fs = True
empatia = False

fitness = 0
accuracy = 0
f1score = 0
precision = 0
recall = 0
mcc = 0
errorRate = 0
totalFeatureSelected = 0

if empatia:
    problema = 'EMPATIA-3'
    print("-------------------------------------------------------------------")
    print(problema)
    print("-------------------------------------------------------------------")
    totalFeature = 57
    individuo = np.ones(57)
    opt = load_parameters("./Problem/EMPATIA/model/")
    
    # LOAD DATA
    data_dir = './Problem/EMPATIA/'
    dataset = EmotionalDataset(data_dir + 'features_47Vol_CSVs',
                                data_dir + 'labels')
    REMOVE_TEST = [4, 5, 14, 32, 39]
    set_vol = dataset.df_data.index.get_level_values('vol_id').unique()
    non_valid_volunteers = [set_vol[i] for i in REMOVE_TEST]
    dataset.filter(filter_idx = non_valid_volunteers)
    loader = BasicLoader(dataset, norm = True)
    loader.dataset.reset_index(drop = True, inplace = True)
    
    fitness, accuracy, f1Score, precision, recall, mcc, errorRate, totalFeatureSelected = get_fitness(loader, individuo, problema, opt)

    
    print(
        f'fitness                 : {str(fitness)}\n'+
        f'accuracy                : {str(accuracy)}\n'+
        f'f-score                 : {str(f1score)}\n'+
        f'precision               : {str(precision)}\n'+
        f'recall                  : {str(recall)}\n'+
        f'mcc                     : {str(mcc)}\n'+
        f'errorRate               : {str(errorRate)}\n'+
        f'total Feature Selected  : {str(totalFeatureSelected)}')
    
if fs:
    # instancias = ['ionosphere','sonar','Immunotherapy','Divorce','wdbc','breast-cancer-wisconsin','only_clinic']
    instancias = ['nefrologia','only_clinic']
    
    clasificadores = ['KNN','RandomForest','Xgboost']
    
    for archivo in instancias:
    
        print("---------------------------------------------------------")
        instancia = FeatureSelection(archivo)
        print("---------------------------------------------------------")
        # print(len(instancia.getDatos().columns))

        individuo = np.ones(instancia.getTotalFeature())
        # individuo = np.array([0,1,1,0,1,1,1,0,0,0,0 ,1 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,1 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,1 ,1 ,1 ,0 ,1 ,0 ,1 ,1 ,0 ,0 ,1 ,0 ,1 ,1 ,0 ,0 ,0 ,1 ,1 ,1 ,0 ,1 ,0 ,1 ,0 ,1 ,1 ,1 ,0 ,1 ,1 ,0 ,1 ,0 ,0 ])
        # [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62]
        # [0,1,1,0,1,1,1,0,0,0,0 ,1 ,0 ,1 ,1 ,1 ,1 ,1 ,1 ,0 ,1 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,1 ,1 ,1 ,0 ,1 ,0 ,1 ,1 ,0 ,0 ,1 ,0 ,1 ,1 ,0 ,0 ,0 ,1 ,1 ,1 ,0 ,1 ,0 ,1 ,0 ,1 ,1 ,1 ,0 ,1 ,1 ,0 ,1 ,0 ,0 ]
        # individuo = np.random.randint(low=0, high=2, size = (len(instancia.getDatos().columns)))
        
        seleccion = np.where(individuo == 1)[0]
        # print(individuo)
        # print(seleccion)

        for clasificador in clasificadores:
        
            fitness, accuracy, f1score, precision, recall, mcc, errorRate, totalFeatureSelected = instancia.fitness(seleccion, clasificador, "k:5")

            print(
                f'--------------------------------------------\n'+
                f'clasificador            : {clasificador}\n'+
                f'fitness                 : {str(fitness)}\n'+
                f'accuracy                : {str(accuracy)}\n'+
                f'f-score                 : {str(f1score)}\n'+
                f'precision               : {str(precision)}\n'+
                f'recall                  : {str(recall)}\n'+
                f'mcc                     : {str(mcc)}\n'+
                f'errorRate               : {str(errorRate)}\n'+
                f'total Feature Selected  : {str(totalFeatureSelected)}\n'+
                f'--------------------------------------------')