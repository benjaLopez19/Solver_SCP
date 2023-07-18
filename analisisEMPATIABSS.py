import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

from util import util
from BD.sqlite import BD

bd = BD()

instancias = bd.obtenerInstancias(f'''
                                  "EMPATIA-9","EMPATIA-10","EMPATIA-11","EMPATIA-12"
                                  ''')
binarizaciones = ["S4-STD","V4-STD","X4-STD","Z4-STD","S4-COM","V4-COM","X4-COM","Z4-COM","S4-ELIT","V4-ELIT","X4-ELIT","Z4-ELIT"]

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
    
    for bss in binarizaciones:
    
        blob = bd.obtenerMejoresArchivosconBSS(instancia[1],"",bss)
        
        archivoResumenFitness = open(f'{dirResultado}resumen_fitness_{instancia[1]}_{bss}.csv', 'w')
        
        for d in blob:
            
            nombreArchivo = d[5]
            archivo = d[6]
            mh = d[1]
            problema = d[4]
            
            direccionDestiono = './Resultados/Transitorio/'+nombreArchivo+"_"+bss+'.csv'
            
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
            
            fitnessWOA = 0
            timeWOA = 0
            xplWOA = 0
            xptWOA = 0
            tfsWOA = 0
            accuracyWOA = 0
            f1ScoreWOA = 0
            precisionWOA = 0
            recallWOA = 0
            mccWOA = 0
            errorRateWOA = 0
            
            
            ultimo = len(iteraciones) - 1
            
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
                
                fitnessWOA = fitness[ultimo]
                timeWOA = np.round(np.sum(time),3)
                xplWOA = np.round(np.mean(xpl), decimals=2)
                xptWOA = np.round(np.mean(xpt), decimals=2)
                tfsWOA = int(tfs[ultimo])
                accuracyWOA = accuracy[ultimo]
                f1ScoreWOA = f1Score[ultimo]
                precisionWOA = precision[ultimo]
                recallWOA = recall[ultimo]
                mccWOA = mcc[ultimo]
                errorRateWOA = errorRate[ultimo]
                
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
            ax.set_title(f'Diversity {mh} - {problema} - {bss}')
            ax.set_ylabel("Diversity")
            ax.set_xlabel("Iteration")
            plt.savefig(f'{dirResultado}/Graficos/Diversity_{mh}_{problema}_{bss}.pdf')
            plt.close('all')
            print(f'Grafico de diversidad realizado {mh} {problema} {bss} ')
            
            figPER, axPER = plt.subplots()
            axPER.plot(iteraciones, xpl, color="r", label=r"$\overline{XPL}$"+": "+str(np.round(np.mean(xpl), decimals=2))+"%")
            axPER.plot(iteraciones, xpt, color="b", label=r"$\overline{XPT}$"+": "+str(np.round(np.mean(xpt), decimals=2))+"%")
            axPER.set_title(f'XPL% - XPT% {mh} - {problema} - {bss}')
            axPER.set_ylabel("Percentage")
            axPER.set_xlabel("Iteration")
            axPER.legend(loc = 'upper right')
            plt.savefig(f'{dirResultado}/Graficos/Percentage_{mh}_{problema}_{bss}.pdf')
            plt.close('all')
            print(f'Grafico de exploracion y explotacion realizado para {mh}, problema: {problema} {bss}')
            
            
            
            os.remove('./Resultados/Transitorio/'+nombreArchivo+"_"+bss+'.csv')
        
        print(bestFitnessGWO)
        # archivoResumenFitness.write(f'''fitness,{np.min(bestFitnessGWO)},{np.round(np.average(bestFitnessGWO),3)},{np.round(np.std(bestFitnessGWO),3)},{np.min(bestFitnessPSA)},{np.round(np.average(bestFitnessPSA),3)},{np.round(np.std(bestFitnessPSA),3)},{np.min(bestFitnessSCA)},{np.round(np.average(bestFitnessSCA),3)},{np.round(np.std(bestFitnessSCA),3)},{np.min(bestFitnessWOA)},{np.round(np.average(bestFitnessWOA),3)},{np.round(np.std(bestFitnessWOA),3)} \n''')
        # archivoResumenFitness.write(f'''MCC,{np.max(bestMCCGWO)},{np.round(np.average(bestMCCGWO),3)},{np.round(np.std(bestMCCGWO),3)},{np.max(bestMCCPSA)},{np.round(np.average(bestMCCPSA),3)},{np.round(np.std(bestMCCPSA),3)},{np.max(bestMCCSCA)},{np.round(np.average(bestMCCSCA),3)},{np.round(np.std(bestMCCSCA),3)},{np.max(bestMCCWOA)},{np.round(np.average(bestMCCWOA),3)},{np.round(np.std(bestMCCWOA),3)} \n''')
        # archivoResumenFitness.write(f'''accuracy,{np.max(bestAccuracyGWO)},{np.round(np.average(bestAccuracyGWO),3)},{np.round(np.std(bestAccuracyGWO),3)},{np.max(bestAccuracyPSA)},{np.round(np.average(bestAccuracyPSA),3)},{np.round(np.std(bestAccuracyPSA),3)},{np.max(bestAccuracySCA)},{np.round(np.average(bestAccuracySCA),3)},{np.round(np.std(bestAccuracySCA),3)},{np.max(bestAccuracyWOA)},{np.round(np.average(bestAccuracyWOA),3)},{np.round(np.std(bestAccuracyWOA),3)} \n''')
        # archivoResumenFitness.write(f'''error rate,{np.min(bestErrorRateGWO)},{np.round(np.average(bestErrorRateGWO),3)},{np.round(np.std(bestErrorRateGWO),3)},{np.min(bestErrorRatePSA)},{np.round(np.average(bestErrorRatePSA),3)},{np.round(np.std(bestErrorRatePSA),3)},{np.min(bestErrorRateSCA)},{np.round(np.average(bestErrorRateSCA),3)},{np.round(np.std(bestErrorRateSCA),3)},{np.min(bestErrorRateWOA)},{np.round(np.average(bestErrorRateWOA),3)},{np.round(np.std(bestErrorRateWOA),3)} \n''')
        # archivoResumenFitness.write(f'''f-score,{np.max(bestFscoreGWO)},{np.round(np.average(bestFscoreGWO),3)},{np.round(np.std(bestFscoreGWO),3)},{np.max(bestFscorePSA)},{np.round(np.average(bestFscorePSA),3)},{np.round(np.std(bestFscorePSA),3)},{np.max(bestFscoreSCA)},{np.round(np.average(bestFscoreSCA),3)},{np.round(np.std(bestFscoreSCA),3)},{np.max(bestFscoreWOA)},{np.round(np.average(bestFscoreWOA),3)},{np.round(np.std(bestFscoreWOA),3)} \n''')
        # archivoResumenFitness.write(f'''precision,{np.max(bestPrecisionGWO)},{np.round(np.average(bestPrecisionGWO),3)},{np.round(np.std(bestPrecisionGWO),3)},{np.max(bestPrecisionPSA)},{np.round(np.average(bestPrecisionPSA),3)},{np.round(np.std(bestPrecisionPSA),3)},{np.max(bestPrecisionSCA)},{np.round(np.average(bestPrecisionSCA),3)},{np.round(np.std(bestPrecisionSCA),3)},{np.max(bestPrecisionWOA)},{np.round(np.average(bestPrecisionWOA),3)},{np.round(np.std(bestPrecisionWOA),3)} \n''')
        # archivoResumenFitness.write(f'''recall,{np.max(bestRecallGWO)},{np.round(np.average(bestRecallGWO),3)},{np.round(np.std(bestRecallGWO),3)},{np.max(bestRecallPSA)},{np.round(np.average(bestRecallPSA),3)},{np.round(np.std(bestRecallPSA),3)},{np.max(bestRecallSCA)},{np.round(np.average(bestRecallSCA),3)},{np.round(np.std(bestRecallSCA),3)},{np.max(bestRecallWOA)},{np.round(np.average(bestRecallWOA),3)},{np.round(np.std(bestRecallWOA),3)} \n''')
        # archivoResumenFitness.write(f'''total feature selected,-,{np.round(np.average(bestTfsGWO),3)},{np.round(np.std(bestTfsGWO),3)},-,{np.round(np.average(bestTfsPSA),3)},{np.round(np.std(bestTfsPSA),3)},-,{np.round(np.average(bestTfsSCA),3)},{np.round(np.std(bestTfsSCA),3)},-,{np.round(np.average(bestTfsWOA),3)},{np.round(np.std(bestTfsWOA),3)} \n''')
        # archivoResumenFitness.write(f'''time,{np.min(bestTimeGWO)},{np.round(np.average(bestTimeGWO),3)},{np.round(np.std(bestTimeGWO),3)},{np.min(bestTimePSA)},{np.round(np.average(bestTimePSA),3)},{np.round(np.std(bestTimePSA),3)},{np.min(bestTimeSCA)},{np.round(np.average(bestTimeSCA),3)},{np.round(np.std(bestTimeSCA),3)},{np.min(bestTimeWOA)},{np.round(np.average(bestTimeWOA),3)},{np.round(np.std(bestTimeWOA),3)} \n''')
            
        archivoResumenFitness.write(f'''{fitnessWOA}\n''')
        archivoResumenFitness.write(f'''{mccWOA}\n''')
        archivoResumenFitness.write(f'''{accuracyWOA}\n''')
        archivoResumenFitness.write(f'''{errorRateWOA}\n''')
        archivoResumenFitness.write(f'''{f1ScoreWOA}\n''')
        archivoResumenFitness.write(f'''{precisionWOA}\n''')
        archivoResumenFitness.write(f'''{recallWOA}\n''')
        archivoResumenFitness.write(f'''{tfsWOA}\n''')
        archivoResumenFitness.write(f'''{timeWOA}\n''')
        print("------------------------------------------------------------------------------------------------------------")
        figPER, axPER = plt.subplots()
        # axPER.plot(iteraciones, bestFitnessGWO, color="r", label="GWO")
        # axPER.plot(iteraciones, bestFitnessSCA, color="b", label="SCA")
        # axPER.plot(iteraciones, bestFitnessPSA, color="g", label="PSA")
        axPER.plot(iteraciones, bestFitnessWOA, color="y", label="WOA")
        # axPER.plot(iteraciones, bestFitnessMFO, color="m", label="MFO")
        axPER.set_title(f'Coverage - {problema} - {bss}')
        axPER.set_ylabel("Fitness")
        axPER.set_xlabel("Iteration")
        axPER.legend(loc = 'upper right')
        plt.savefig(f'{dirResultado}/Best/fitness_{problema}_{bss}.pdf')
        plt.close('all')
        print(f'Grafico de fitness realizado {problema} ')
        
        figPER, axPER = plt.subplots()
        # axPER.plot(iteraciones, bestTimeGWO, color="r", label="GWO")
        # axPER.plot(iteraciones, bestTimeSCA, color="b", label="SCA")
        # axPER.plot(iteraciones, bestTimePSA, color="g", label="PSA")
        axPER.plot(iteraciones, bestTimeWOA, color="y", label="WOA")
        # axPER.plot(iteraciones, bestTimeMFO, color="m", label="MFO")
        axPER.set_title(f'Time (s) - {problema} - {bss}')
        axPER.set_ylabel("Time (s)")
        axPER.set_xlabel("Iteration")
        axPER.legend(loc = 'upper right')
        plt.savefig(f'{dirResultado}/Best/time_{problema}_{bss}.pdf')
        plt.close('all')
        print(f'Grafico de time realizado {problema} ')
        
        figPER, axPER = plt.subplots()
        # axPER.plot(iteraciones, bestTfsGWO, color="r", label="GWO")
        # axPER.plot(iteraciones, bestTfsSCA, color="b", label="SCA")
        # axPER.plot(iteraciones, bestTfsPSA, color="g", label="PSA")
        axPER.plot(iteraciones, bestTfsWOA, color="y", label="WOA")
        # axPER.plot(iteraciones, bestTfsMFO, color="m", label="MFO")
        axPER.set_title(f'Total feature selected - {problema} - {bss}')
        axPER.set_ylabel("Total feature selected")
        axPER.set_xlabel("Iteration")
        axPER.legend(loc = 'upper right')
        plt.savefig(f'{dirResultado}/Best/tfs_{problema}_{bss}.pdf')
        plt.close('all')
        print(f'Grafico de total feature selected realizado {problema} ')
        
        figPER, axPER = plt.subplots()
        # axPER.plot(iteraciones, bestAccuracyGWO, color="r", label="GWO")
        # axPER.plot(iteraciones, bestAccuracySCA, color="b", label="SCA")
        # axPER.plot(iteraciones, bestAccuracyPSA, color="g", label="PSA")
        axPER.plot(iteraciones, bestAccuracyWOA, color="y", label="WOA")
        # axPER.plot(iteraciones, bestAccuracyMFO, color="m", label="MFO")
        axPER.set_title(f'Accuracy - {problema} - {bss}')
        axPER.set_ylabel("Accuracy")
        axPER.set_xlabel("Iteration")
        axPER.legend(loc = 'lower right')
        plt.savefig(f'{dirResultado}/Best/accuracy_{problema}_{bss}.pdf')
        plt.close('all')
        print(f'Grafico de accuracy realizado {problema} ')
        
        figPER, axPER = plt.subplots()
        # axPER.plot(iteraciones, bestFscoreGWO, color="r", label="GWO")
        # axPER.plot(iteraciones, bestFscoreSCA, color="b", label="SCA")
        # axPER.plot(iteraciones, bestFscorePSA, color="g", label="PSA")
        axPER.plot(iteraciones, bestFscoreWOA, color="y", label="WOA")
        # axPER.plot(iteraciones, bestFscoreMFO, color="m", label="MFO")
        axPER.set_title(f'f-score - {problema} - {bss}')
        axPER.set_ylabel("f-score")
        axPER.set_xlabel("Iteration")
        axPER.legend(loc = 'lower right')
        plt.savefig(f'{dirResultado}/Best/fscore_{problema}_{bss}.pdf')
        plt.close('all')
        print(f'Grafico de f-score realizado {problema} ')
        
        figPER, axPER = plt.subplots()
        # axPER.plot(iteraciones, bestPrecisionGWO, color="r", label="GWO")
        # axPER.plot(iteraciones, bestPrecisionSCA, color="b", label="SCA")
        # axPER.plot(iteraciones, bestPrecisionPSA, color="g", label="PSA")
        axPER.plot(iteraciones, bestPrecisionWOA, color="y", label="WOA")
        # axPER.plot(iteraciones, bestPrecisionMFO, color="m", label="MFO")
        axPER.set_title(f'Precision - {problema} - {bss}')
        axPER.set_ylabel("Precision")
        axPER.set_xlabel("Iteration")
        axPER.legend(loc = 'lower right')
        plt.savefig(f'{dirResultado}/Best/precision_{problema}_{bss}.pdf')
        plt.close('all')
        print(f'Grafico de precision realizado {problema} ')
        
        figPER, axPER = plt.subplots()
        # axPER.plot(iteraciones, bestRecallGWO, color="r", label="GWO")
        # axPER.plot(iteraciones, bestRecallSCA, color="b", label="SCA")
        # axPER.plot(iteraciones, bestRecallPSA, color="g", label="PSA")
        axPER.plot(iteraciones, bestRecallWOA, color="y", label="WOA")
        # axPER.plot(iteraciones, bestRecallMFO, color="m", label="MFO")
        axPER.set_title(f'Recall - {problema} - {bss}')
        axPER.set_ylabel("Recall")
        axPER.set_xlabel("Iteration")
        axPER.legend(loc = 'lower right')
        plt.savefig(f'{dirResultado}/Best/recall_{problema}_{bss}.pdf')
        plt.close('all')
        print(f'Grafico de recall realizado {problema} ')
        
        figPER, axPER = plt.subplots()
        # axPER.plot(iteraciones, bestMCCGWO, color="r", label="GWO")
        # axPER.plot(iteraciones, bestMCCSCA, color="b", label="SCA")
        # axPER.plot(iteraciones, bestMCCPSA, color="g", label="PSA")
        axPER.plot(iteraciones, bestMCCWOA, color="y", label="WOA")
        # axPER.plot(iteraciones, bestMCCMFO, color="m", label="MFO")
        axPER.set_title(f'Matthew’s correlation coefficients \n {problema} - {bss}')
        axPER.set_ylabel("Matthew’s correlation coefficients")
        axPER.set_xlabel("Iteration")
        axPER.legend(loc = 'lower right')
        plt.savefig(f'{dirResultado}/Best/mcc_{problema}_{bss}.pdf')
        plt.close('all')
        print(f'Grafico de MCC realizado {problema} ')
        
        figPER, axPER = plt.subplots()
        # axPER.plot(iteraciones, bestErrorRateGWO, color="r", label="GWO")
        # axPER.plot(iteraciones, bestErrorRateSCA, color="b", label="SCA")
        # axPER.plot(iteraciones, bestErrorRatePSA, color="g", label="PSA")
        axPER.plot(iteraciones, bestErrorRateWOA, color="y", label="WOA")
        # axPER.plot(iteraciones, bestErrorRateMFO, color="m", label="MFO")
        axPER.set_title(f'Error Rate - {problema} - {bss}')
        axPER.set_ylabel("Error Rate")
        axPER.set_xlabel("Iteration")
        axPER.legend(loc = 'upper right')
        plt.savefig(f'{dirResultado}/Best/errorRate_{problema}_{bss}.pdf')
        plt.close('all')
        print(f'Grafico de error rate realizado {problema} ')
        
        archivoResumenFitness.close()