import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

from util import util
from BD.sqlite import BD

bd = BD()

instancias = bd.obtenerInstancias(f'''
                                  "nefrologia","only_clinic"
                                  ''')
clasificadores = ["KNN","RandomForest","Xgboost"]

dirResultado = './Resultados/'

# --------------------------------------------------------------------------------------- #
    
bestFitnessSCA = []
bestFitnessGWO = []
bestFitnessWOA = []
bestFitnessPSA = []
bestFitnessMFO = []

bestTimeSCA = []
bestTimeGWO = []
bestTimeWOA = []
bestTimePSA = []
bestTimeMFO = []

bestTfsSCA = []
bestTfsGWO = []
bestTfsWOA = []
bestTfsPSA = []
bestTfsMFO = []

bestAccuracySCA = []
bestAccuracyGWO = []
bestAccuracyWOA = []
bestAccuracyPSA = []
bestAccuracyMFO = []


bestFscoreSCA = []
bestFscoreGWO = []
bestFscoreWOA = []
bestFscorePSA = []
bestFscoreMFO = []

bestPrecisionSCA = []
bestPrecisionGWO = []
bestPrecisionWOA = []
bestPrecisionPSA = []
bestPrecisionMFO = []


bestRecallSCA = []
bestRecallGWO = []
bestRecallWOA = []
bestRecallPSA = []
bestRecallMFO = []

bestMCCSCA = []
bestMCCGWO = []
bestMCCWOA = []
bestMCCPSA = []
bestMCCMFO = []

bestErrorRateSCA = []
bestErrorRateGWO = []
bestErrorRateWOA = []
bestErrorRatePSA = []
bestErrorRateMFO = []

bestDIVSCA = []
bestDIVGWO = []
bestDIVWOA = []
bestDIVPSA = []
bestDIVMFO = []

print(instancias)
for instancia in instancias:
    
    print(instancia)
    
    for clasificador in clasificadores:
    
        blob = bd.obtenerMejoresArchivosconClasificador(instancia[1],"",clasificador)
        
        
        
        for d in blob:
            
            nombreArchivo = d[5]
            archivo = d[6]
            mh = d[1]
            problema = d[4]
            
            direccionDestiono = './Resultados/Transitorio/'+nombreArchivo+"_"+clasificador+'.csv'
            
            print(nombreArchivo)
            
            util.writeTofile(archivo,direccionDestiono)
        
            data = pd.read_csv(direccionDestiono)
            
            
            iteraciones = data['iter']
            fitness     = data['fitness']
            time        = data['time']
            accuracy    = data['accuracy']
            f1Score     = data['f1-score']
            precision   = data['precision']
            recall      = data['recall']
            mcc         = data['mcc']
            errorRate   = data['errorRate']
            tfs         = data['TFS']
            xpl         = data['XPL']
            xpt         = data['XPT']
            div         = data['DIV']
            
            if mh == 'PSA':
                bestFitnessPSA      = fitness
                bestTimePSA         = time
                bestTfsPSA          = tfs
                bestAccuracyPSA     = accuracy
                bestFscorePSA       = f1Score
                bestPrecisionPSA    = precision
                bestRecallPSA       = recall
                bestMCCPSA          = mcc
                bestErrorRatePSA    = errorRate
                bestDIVPSA          = div
            if mh == 'SCA':
                bestFitnessSCA      = fitness
                bestTimeSCA         = time
                bestTfsSCA          = tfs
                bestAccuracySCA     = accuracy
                bestFscoreSCA       = f1Score
                bestPrecisionSCA    = precision
                bestRecallSCA       = recall
                bestMCCSCA          = mcc
                bestErrorRateSCA    = errorRate
                bestDIVSCA          = div
            if mh == 'GWO':
                bestFitnessGWO      = fitness
                bestTimeGWO         = time
                bestTfsGWO          = tfs
                bestAccuracyGWO     = accuracy
                bestFscoreGWO       = f1Score
                bestPrecisionGWO    = precision
                bestRecallGWO       = recall
                bestMCCGWO          = mcc
                bestErrorRateGWO    = errorRate
                bestDIVGWO          = div
            if mh == 'WOA':
                bestFitnessWOA      = fitness
                bestTimeWOA         = time
                bestTfsWOA          = tfs
                bestAccuracyWOA     = accuracy
                bestFscoreWOA       = f1Score
                bestPrecisionWOA    = precision
                bestRecallWOA       = recall
                bestMCCWOA          = mcc
                bestErrorRateWOA    = errorRate
                bestDIVWOA          = div
                
            if mh == 'MFO':
                bestFitnessMFO      = fitness
                bestTimeMFO         = time
                bestTfsMFO          = tfs
                bestAccuracyMFO     = accuracy
                bestFscoreMFO       = f1Score
                bestPrecisionMFO    = precision
                bestRecallMFO       = recall
                bestMCCMFO          = mcc
                bestErrorRateMFO    = errorRate
                bestDIVMFO          = div
                
            fig , ax = plt.subplots()
            ax.plot(iteraciones,div)
            ax.set_title(f'Diversity {mh} - {problema} - {clasificador}')
            ax.set_ylabel("Diversity")
            ax.set_xlabel("Iteration")
            plt.savefig(f'{dirResultado}/Graficos/Diversity_{mh}_{problema}_{clasificador}.pdf')
            plt.close('all')
            print(f'Grafico de diversidad realizado {mh} {problema} {clasificador} ')
            
            figPER, axPER = plt.subplots()
            axPER.plot(iteraciones, xpl, color="r", label=r"$\overline{XPL}$"+": "+str(np.round(np.mean(xpl), decimals=2))+"%")
            axPER.plot(iteraciones, xpt, color="b", label=r"$\overline{XPT}$"+": "+str(np.round(np.mean(xpt), decimals=2))+"%")
            axPER.set_title(f'XPL% - XPT% {mh} - {problema} - {clasificador}')
            axPER.set_ylabel("Percentage")
            axPER.set_xlabel("Iteration")
            axPER.legend(loc = 'upper right')
            plt.savefig(f'{dirResultado}/Graficos/Percentage_{mh}_{problema}_{clasificador}.pdf')
            plt.close('all')
            print(f'Grafico de exploracion y explotacion realizado para {mh}, problema: {problema} {clasificador}')
            
            
            
            os.remove('./Resultados/Transitorio/'+nombreArchivo+"_"+clasificador+'.csv')
            
        print("------------------------------------------------------------------------------------------------------------")
        figPER, axPER = plt.subplots()
        axPER.plot(iteraciones, bestFitnessGWO, color="r", label="GWO")
        axPER.plot(iteraciones, bestFitnessSCA, color="b", label="SCA")
        axPER.plot(iteraciones, bestFitnessPSA, color="g", label="PSA")
        axPER.plot(iteraciones, bestFitnessWOA, color="y", label="WOA")
        # axPER.plot(iteraciones, bestFitnessMFO, color="m", label="MFO")
        axPER.set_title(f'Coverage - {problema} - {clasificador}')
        axPER.set_ylabel("Fitness")
        axPER.set_xlabel("Iteration")
        axPER.legend(loc = 'upper right')
        plt.savefig(f'{dirResultado}/Best/fitness_{problema}_{clasificador}.pdf')
        plt.close('all')
        print(f'Grafico de fitness realizado {problema} ')
        
        figPER, axPER = plt.subplots()
        axPER.plot(iteraciones, bestTimeGWO, color="r", label="GWO")
        axPER.plot(iteraciones, bestTimeSCA, color="b", label="SCA")
        axPER.plot(iteraciones, bestTimePSA, color="g", label="PSA")
        axPER.plot(iteraciones, bestTimeWOA, color="y", label="WOA")
        # axPER.plot(iteraciones, bestTimeMFO, color="m", label="MFO")
        axPER.set_title(f'Time (s) - {problema} - {clasificador}')
        axPER.set_ylabel("Time (s)")
        axPER.set_xlabel("Iteration")
        axPER.legend(loc = 'upper right')
        plt.savefig(f'{dirResultado}/Best/time_{problema}_{clasificador}.pdf')
        plt.close('all')
        print(f'Grafico de time realizado {problema} ')
        
        figPER, axPER = plt.subplots()
        axPER.plot(iteraciones, bestTfsGWO, color="r", label="GWO")
        axPER.plot(iteraciones, bestTfsSCA, color="b", label="SCA")
        axPER.plot(iteraciones, bestTfsPSA, color="g", label="PSA")
        axPER.plot(iteraciones, bestTfsWOA, color="y", label="WOA")
        # axPER.plot(iteraciones, bestTfsMFO, color="m", label="MFO")
        axPER.set_title(f'Total feature selected - {problema} - {clasificador}')
        axPER.set_ylabel("Total feature selected")
        axPER.set_xlabel("Iteration")
        axPER.legend(loc = 'upper right')
        plt.savefig(f'{dirResultado}/Best/tfs_{problema}_{clasificador}.pdf')
        plt.close('all')
        print(f'Grafico de total feature selected realizado {problema} ')
        
        figPER, axPER = plt.subplots()
        axPER.plot(iteraciones, bestAccuracyGWO, color="r", label="GWO")
        axPER.plot(iteraciones, bestAccuracySCA, color="b", label="SCA")
        axPER.plot(iteraciones, bestAccuracyPSA, color="g", label="PSA")
        axPER.plot(iteraciones, bestAccuracyWOA, color="y", label="WOA")
        # axPER.plot(iteraciones, bestAccuracyMFO, color="m", label="MFO")
        axPER.set_title(f'Accuracy - {problema} - {clasificador}')
        axPER.set_ylabel("Accuracy")
        axPER.set_xlabel("Iteration")
        axPER.legend(loc = 'lower right')
        plt.savefig(f'{dirResultado}/Best/accuracy_{problema}_{clasificador}.pdf')
        plt.close('all')
        print(f'Grafico de accuracy realizado {problema} ')
        
        figPER, axPER = plt.subplots()
        axPER.plot(iteraciones, bestFscoreGWO, color="r", label="GWO")
        axPER.plot(iteraciones, bestFscoreSCA, color="b", label="SCA")
        axPER.plot(iteraciones, bestFscorePSA, color="g", label="PSA")
        axPER.plot(iteraciones, bestFscoreWOA, color="y", label="WOA")
        # axPER.plot(iteraciones, bestFscoreMFO, color="m", label="MFO")
        axPER.set_title(f'f-score - {problema} - {clasificador}')
        axPER.set_ylabel("f-score")
        axPER.set_xlabel("Iteration")
        axPER.legend(loc = 'lower right')
        plt.savefig(f'{dirResultado}/Best/fscore_{problema}_{clasificador}.pdf')
        plt.close('all')
        print(f'Grafico de f-score realizado {problema} ')
        
        figPER, axPER = plt.subplots()
        axPER.plot(iteraciones, bestPrecisionGWO, color="r", label="GWO")
        axPER.plot(iteraciones, bestPrecisionSCA, color="b", label="SCA")
        axPER.plot(iteraciones, bestPrecisionPSA, color="g", label="PSA")
        axPER.plot(iteraciones, bestPrecisionWOA, color="y", label="WOA")
        # axPER.plot(iteraciones, bestPrecisionMFO, color="m", label="MFO")
        axPER.set_title(f'Precision - {problema} - {clasificador}')
        axPER.set_ylabel("Precision")
        axPER.set_xlabel("Iteration")
        axPER.legend(loc = 'lower right')
        plt.savefig(f'{dirResultado}/Best/precision_{problema}_{clasificador}.pdf')
        plt.close('all')
        print(f'Grafico de precision realizado {problema} ')
        
        figPER, axPER = plt.subplots()
        axPER.plot(iteraciones, bestRecallGWO, color="r", label="GWO")
        axPER.plot(iteraciones, bestRecallSCA, color="b", label="SCA")
        axPER.plot(iteraciones, bestRecallPSA, color="g", label="PSA")
        axPER.plot(iteraciones, bestRecallWOA, color="y", label="WOA")
        # axPER.plot(iteraciones, bestRecallMFO, color="m", label="MFO")
        axPER.set_title(f'Recall - {problema} - {clasificador}')
        axPER.set_ylabel("Recall")
        axPER.set_xlabel("Iteration")
        axPER.legend(loc = 'lower right')
        plt.savefig(f'{dirResultado}/Best/recall_{problema}_{clasificador}.pdf')
        plt.close('all')
        print(f'Grafico de recall realizado {problema} ')
        
        figPER, axPER = plt.subplots()
        axPER.plot(iteraciones, bestMCCGWO, color="r", label="GWO")
        axPER.plot(iteraciones, bestMCCSCA, color="b", label="SCA")
        axPER.plot(iteraciones, bestMCCPSA, color="g", label="PSA")
        axPER.plot(iteraciones, bestMCCWOA, color="y", label="WOA")
        # axPER.plot(iteraciones, bestMCCMFO, color="m", label="MFO")
        axPER.set_title(f'Matthew’s correlation coefficients \n {problema} - {clasificador}')
        axPER.set_ylabel("Matthew’s correlation coefficients")
        axPER.set_xlabel("Iteration")
        axPER.legend(loc = 'lower right')
        plt.savefig(f'{dirResultado}/Best/mcc_{problema}_{clasificador}.pdf')
        plt.close('all')
        print(f'Grafico de MCC realizado {problema} ')
        
        figPER, axPER = plt.subplots()
        axPER.plot(iteraciones, bestErrorRateGWO, color="r", label="GWO")
        axPER.plot(iteraciones, bestErrorRateSCA, color="b", label="SCA")
        axPER.plot(iteraciones, bestErrorRatePSA, color="g", label="PSA")
        axPER.plot(iteraciones, bestErrorRateWOA, color="y", label="WOA")
        # axPER.plot(iteraciones, bestErrorRateMFO, color="m", label="MFO")
        axPER.set_title(f'Error Rate - {problema} - {clasificador}')
        axPER.set_ylabel("Error Rate")
        axPER.set_xlabel("Iteration")
        axPER.legend(loc = 'upper right')
        plt.savefig(f'{dirResultado}/Best/errorRate_{problema}_{clasificador}.pdf')
        plt.close('all')
        print(f'Grafico de error rate realizado {problema} ')