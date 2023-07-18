
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

from util import util
from BD.sqlite import BD


dirResultado = './Resultados/'

graficos = True

bd = BD()

instancias = bd.obtenerInstancias(f'''
                                  "EMPATIA","EMPATIA-2","EMPATIA-3","EMPATIA-4"
                                  ''')
print(instancias)
for instancia in instancias:
    
    print(instancia)
    
    problema = instancia[1]
    
    archivoResumenFitness = open(f'{dirResultado}resumen_fitness_{problema}_Q-Learning.csv', 'w')
    archivoResumenPercentage = open(f'{dirResultado}resumen_percentage_{problema}_Q-Learning.csv', 'w')

    archivoResumenFitness.write("metric,best,avg,dev-std,best,avg,dev-std,best,avg,dev-std,best,avg,dev-std\n")
    archivoResumenPercentage.write("percentage, avg. XPL%, avg. XPT%, avg. XPL%, avg. XPT%, avg. XPL%, avg. XPT%, avg. XPL%, avg. XPT%\n")

    blob = bd.obtenerArchivos(instancia[1])
    problema = instancia[1]
    
    corrida = 1

    
    archivoFitness = open(f'{dirResultado}fitness_'+problema+'.csv', 'w')
    archivoFitness.write('MH,FITNESS\n')
    
    archivoAccuracy = open(f'{dirResultado}accuracy_'+problema+'.csv', 'w')
    archivoAccuracy.write('MH,ACCURACY\n')
    
    archivoMCC = open(f'{dirResultado}mcc_'+problema+'.csv', 'w')
    archivoMCC.write('MH,MCC\n')

    fitnessSCA = [] 
    fitnessGWO = [] 
    fitnessWOA = [] 
    fitnessPSA = []

    timeSCA = []
    timeGWO = []
    timeWOA = []
    timePSA = []

    xplSCA = [] 
    xplGWO = [] 
    xplWOA = [] 
    xplPSA = []

    xptSCA = []
    xptGWO = []
    xptWOA = []
    xptPSA = []
    
    tfsSCA = []
    tfsGWO = []
    tfsWOA = []
    tfsPSA = []
    
    accuracySCA = []
    accuracyGWO = []
    accuracyWOA = []
    accuracyPSA = []
    
    f1ScoreSCA = []
    f1ScoreGWO = []
    f1ScoreWOA = []
    f1ScorePSA = []
    
    precisionSCA = []
    precisionGWO = []
    precisionWOA = []
    precisionPSA = []
    
    recallSCA = []
    recallGWO = []
    recallWOA = []
    recallPSA = []
    
    mccSCA = []
    mccGWO = []
    mccWOA = []
    mccPSA = []
    
    errorRateSCA = []
    errorRateGWO = []
    errorRateWOA = []
    errorRatePSA = []
    
    divSCA = []
    divGWO = []
    divWOA = []
    divPSA = []
    
    # errorRate   = data['errorRate']
    
    # --------------------------------------------------------------------------------------- #
    
    bestFitnessSCA = []
    bestFitnessGWO = []
    bestFitnessWOA = []
    bestFitnessPSA = []

    bestTimeSCA = []
    bestTimeGWO = []
    bestTimeWOA = []
    bestTimePSA = []
    
    bestTfsSCA = []
    bestTfsGWO = []
    bestTfsWOA = []
    bestTfsPSA = []
    
    bestAccuracySCA = []
    bestAccuracyGWO = []
    bestAccuracyWOA = []
    bestAccuracyPSA = []
    
    bestFscoreSCA = []
    bestFscoreGWO = []
    bestFscoreWOA = []
    bestFscorePSA = []
    
    bestPrecisionSCA = []
    bestPrecisionGWO = []
    bestPrecisionWOA = []
    bestPrecisionPSA = []
    
    bestPrecisionSCA = []
    bestPrecisionGWO = []
    bestPrecisionWOA = []
    bestPrecisionPSA = []
    
    bestRecallSCA = []
    bestRecallGWO = []
    bestRecallWOA = []
    bestRecallPSA = []
    
    bestMCCSCA = []
    bestMCCGWO = []
    bestMCCWOA = []
    bestMCCPSA = []
    
    bestErrorRateSCA = []
    bestErrorRateGWO = []
    bestErrorRateWOA = []
    bestErrorRatePSA = []
    
    bestDivSCA = []
    bestDivGWO = []
    bestDivWOA = []
    bestDivPSA = []

    
    for d in blob:

        nombreArchivo = d[0]
        archivo = d[1]

        if len(nombreArchivo.split("_")) > 2:
        
            direccionDestiono = './Resultados/Transitorio/'+nombreArchivo+'.csv'
            # print("-------------------------------------------------------------------------------")
            util.writeTofile(archivo,direccionDestiono)
            
            data = pd.read_csv(direccionDestiono)
            
            mh = nombreArchivo.split('_')[0]
            problem = nombreArchivo.split('_')[1]

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
            
            ultimo = len(iteraciones) - 1
            
            if mh == 'PSA':            
                fitnessPSA.append(fitness[ultimo])
                timePSA.append(np.round(np.sum(time),3))
                xplPSA.append(np.round(np.mean(xpl), decimals=2))
                xptPSA.append(np.round(np.mean(xpt), decimals=2))
                tfsPSA.append(tfs[ultimo])
                accuracyPSA.append(accuracy[ultimo])
                f1ScorePSA.append(f1Score[ultimo])
                precisionPSA.append(precision[ultimo])
                recallPSA.append(recall[ultimo])
                mccPSA.append(mcc[ultimo])
                errorRatePSA.append(errorRate[ultimo])
                
                
                archivoFitness.write(f'PSA,{str(fitness[ultimo])}\n')
                archivoAccuracy.write(f'PSA,{str(accuracy[ultimo])}\n')
                archivoMCC.write(f'PSA,{str(mcc[ultimo])}\n')
            if mh == 'SCA':
                fitnessSCA.append(fitness[ultimo])
                timeSCA.append(np.round(np.sum(time),3))
                xplSCA.append(np.round(np.mean(xpl), decimals=2))
                xptSCA.append(np.round(np.mean(xpt), decimals=2))
                tfsSCA.append(tfs[ultimo])
                accuracySCA.append(accuracy[ultimo])
                f1ScoreSCA.append(f1Score[ultimo])
                precisionSCA.append(precision[ultimo])
                recallSCA.append(recall[ultimo])
                mccSCA.append(mcc[ultimo])
                errorRateSCA.append(errorRate[ultimo])
                
                
                archivoFitness.write(f'SCA,{str(fitness[ultimo])}\n')
                archivoAccuracy.write(f'SCA,{str(accuracy[ultimo])}\n')
                archivoMCC.write(f'SCA,{str(mcc[ultimo])}\n')
            if mh == 'GWO':
                fitnessGWO.append(fitness[ultimo])
                timeGWO.append(np.round(np.sum(time),3))
                xplGWO.append(np.round(np.mean(xpl), decimals=2))
                xptGWO.append(np.round(np.mean(xpt), decimals=2))
                tfsGWO.append(tfs[ultimo])
                accuracyGWO.append(accuracy[ultimo])
                f1ScoreGWO.append(f1Score[ultimo])
                precisionGWO.append(precision[ultimo])
                recallGWO.append(recall[ultimo])
                mccGWO.append(mcc[ultimo])
                errorRateGWO.append(errorRate[ultimo])
                
                
                archivoFitness.write(f'GWO,{str(fitness[ultimo])}\n')
                archivoAccuracy.write(f'GWO,{str(accuracy[ultimo])}\n')
                archivoMCC.write(f'GWO,{str(mcc[ultimo])}\n')
            if mh == 'WOA':
                fitnessWOA.append(fitness[ultimo])
                timeWOA.append(np.round(np.sum(time),3))
                xplWOA.append(np.round(np.mean(xpl), decimals=2))
                xptWOA.append(np.round(np.mean(xpt), decimals=2))
                tfsWOA.append(tfs[ultimo])
                accuracyWOA.append(accuracy[ultimo])
                f1ScoreWOA.append(f1Score[ultimo])
                precisionWOA.append(precision[ultimo])
                recallWOA.append(recall[ultimo])
                mccWOA.append(mcc[ultimo])
                errorRateWOA.append(errorRate[ultimo])
                
                archivoFitness.write(f'WOA,{str(fitness[ultimo])}\n')
                archivoAccuracy.write(f'WOA,{str(accuracy[ultimo])}\n')
                archivoMCC.write(f'WOA,{str(mcc[ultimo])}\n')
            
            os.remove('./Resultados/Transitorio/'+nombreArchivo+'.csv')
    
    # print(fitnessGWO)
    # print(fitnessPSA)
    # print(fitnessSCA)
    # print(fitnessWOA)
     
    # print(f'''
    #     {problem},{np.min(fitnessGWO)},{np.round(np.average(fitnessGWO),3)},{np.round(np.std(fitnessGWO),3)},{np.min(fitnessPSA)},{np.round(np.average(fitnessPSA),3)},{np.round(np.std(fitnessPSA),3)},{np.min(fitnessSCA)},{np.round(np.average(fitnessSCA),3)},{np.round(np.std(fitnessSCA),3)},{np.min(fitnessWOA)},{np.round(np.average(fitnessWOA),3)},{np.round(np.std(fitnessWOA),3)}
    # ''')
    # print(f'''
    #     {problem},{np.min(timeGWO)},{np.round(np.average(timeGWO),3)},{np.round(np.std(timeGWO),3)},{np.min(timePSA)},{np.round(np.average(timePSA),3)},{np.round(np.std(timePSA),3)},{np.min(timeSCA)},{np.round(np.average(timeSCA),3)},{np.round(np.std(timeSCA),3)},{np.min(timeWOA)},{np.round(np.average(timeWOA),3)},{np.round(np.std(timeWOA),3)}
    # ''')
    # print(f'''
    #     {problem},{np.round(np.average(xplGWO),3)},{np.round(np.average(xptGWO),3)},{np.round(np.average(xplPSA),3)},{np.round(np.average(xptPSA),3)},{np.round(np.average(xplSCA),3)},{np.round(np.average(xptSCA),3)},{np.round(np.average(xplWOA),3)},{np.round(np.average(xptWOA),3)}
    # ''')
    
    archivoResumenFitness.write(f'''fitness,{np.min(fitnessGWO)},{np.round(np.average(fitnessGWO),3)},{np.round(np.std(fitnessGWO),3)},{np.min(fitnessPSA)},{np.round(np.average(fitnessPSA),3)},{np.round(np.std(fitnessPSA),3)},{np.min(fitnessSCA)},{np.round(np.average(fitnessSCA),3)},{np.round(np.std(fitnessSCA),3)},{np.min(fitnessWOA)},{np.round(np.average(fitnessWOA),3)},{np.round(np.std(fitnessWOA),3)} \n''')
    archivoResumenFitness.write(f'''MCC,{np.max(mccGWO)},{np.round(np.average(mccGWO),3)},{np.round(np.std(mccGWO),3)},{np.max(mccPSA)},{np.round(np.average(mccPSA),3)},{np.round(np.std(mccPSA),3)},{np.max(mccSCA)},{np.round(np.average(mccSCA),3)},{np.round(np.std(mccSCA),3)},{np.max(mccWOA)},{np.round(np.average(mccWOA),3)},{np.round(np.std(mccWOA),3)} \n''')
    archivoResumenFitness.write(f'''accuracy,{np.max(accuracyGWO)},{np.round(np.average(accuracyGWO),3)},{np.round(np.std(accuracyGWO),3)},{np.max(accuracyPSA)},{np.round(np.average(accuracyPSA),3)},{np.round(np.std(accuracyPSA),3)},{np.max(accuracySCA)},{np.round(np.average(accuracySCA),3)},{np.round(np.std(accuracySCA),3)},{np.max(accuracyWOA)},{np.round(np.average(accuracyWOA),3)},{np.round(np.std(accuracyWOA),3)} \n''')
    archivoResumenFitness.write(f'''error rate,{np.min(errorRateGWO)},{np.round(np.average(errorRateGWO),3)},{np.round(np.std(errorRateGWO),3)},{np.min(errorRatePSA)},{np.round(np.average(errorRatePSA),3)},{np.round(np.std(errorRatePSA),3)},{np.min(errorRateSCA)},{np.round(np.average(errorRateSCA),3)},{np.round(np.std(errorRateSCA),3)},{np.min(errorRateWOA)},{np.round(np.average(errorRateWOA),3)},{np.round(np.std(errorRateWOA),3)} \n''')
    archivoResumenFitness.write(f'''f-score,{np.max(f1ScoreGWO)},{np.round(np.average(f1ScoreGWO),3)},{np.round(np.std(f1ScoreGWO),3)},{np.max(f1ScorePSA)},{np.round(np.average(f1ScorePSA),3)},{np.round(np.std(f1ScorePSA),3)},{np.max(f1ScoreSCA)},{np.round(np.average(f1ScoreSCA),3)},{np.round(np.std(f1ScoreSCA),3)},{np.max(f1ScoreWOA)},{np.round(np.average(f1ScoreWOA),3)},{np.round(np.std(f1ScoreWOA),3)} \n''')
    archivoResumenFitness.write(f'''precision,{np.max(precisionGWO)},{np.round(np.average(precisionGWO),3)},{np.round(np.std(precisionGWO),3)},{np.max(precisionPSA)},{np.round(np.average(precisionPSA),3)},{np.round(np.std(precisionPSA),3)},{np.max(precisionSCA)},{np.round(np.average(precisionSCA),3)},{np.round(np.std(precisionSCA),3)},{np.max(precisionWOA)},{np.round(np.average(precisionWOA),3)},{np.round(np.std(precisionWOA),3)} \n''')
    archivoResumenFitness.write(f'''recall,{np.max(recallGWO)},{np.round(np.average(recallGWO),3)},{np.round(np.std(recallGWO),3)},{np.max(recallPSA)},{np.round(np.average(recallPSA),3)},{np.round(np.std(recallPSA),3)},{np.max(recallSCA)},{np.round(np.average(recallSCA),3)},{np.round(np.std(recallSCA),3)},{np.max(recallWOA)},{np.round(np.average(recallWOA),3)},{np.round(np.std(recallWOA),3)} \n''')
    archivoResumenFitness.write(f'''total feature selected,-,{np.round(np.average(tfsGWO),3)},{np.round(np.std(tfsGWO),3)},-,{np.round(np.average(tfsPSA),3)},{np.round(np.std(tfsPSA),3)},-,{np.round(np.average(tfsSCA),3)},{np.round(np.std(tfsSCA),3)},-,{np.round(np.average(tfsWOA),3)},{np.round(np.std(tfsWOA),3)} \n''')
    archivoResumenFitness.write(f'''time,{np.min(timeGWO)},{np.round(np.average(timeGWO),3)},{np.round(np.std(timeGWO),3)},{np.min(timePSA)},{np.round(np.average(timePSA),3)},{np.round(np.std(timePSA),3)},{np.min(timeSCA)},{np.round(np.average(timeSCA),3)},{np.round(np.std(timeSCA),3)},{np.min(timeWOA)},{np.round(np.average(timeWOA),3)},{np.round(np.std(timeWOA),3)} \n''')
    
    
    
    archivoResumenPercentage.write(f'''{problema},{np.round(np.average(xplGWO),3)},{np.round(np.average(xptGWO),3)},{np.round(np.average(xplPSA),3)},{np.round(np.average(xptPSA),3)},{np.round(np.average(xplSCA),3)},{np.round(np.average(xptSCA),3)},{np.round(np.average(xplWOA),3)},{np.round(np.average(xptWOA),3)} \n''')
    
    blob = bd.obtenerMejoresArchivos(instancia[1],'Q-Learning')

    for d in blob:

        nombreArchivo = d[4]
        archivo = d[5]

        direccionDestiono = './Resultados/Transitorio/'+nombreArchivo+'.csv'
        util.writeTofile(archivo,direccionDestiono)
        
        data = pd.read_csv(direccionDestiono)
        
        mh = nombreArchivo.split('_')[0]
        problem = nombreArchivo.split('_')[1]

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
            
        if graficos:
            
            figPER, axPER = plt.subplots()
            axPER.plot(iteraciones, xpl, color="r", label=r"$\overline{XPL}$"+": "+str(np.round(np.mean(xpl), decimals=2))+"%")
            axPER.plot(iteraciones, xpt, color="b", label=r"$\overline{XPT}$"+": "+str(np.round(np.mean(xpt), decimals=2))+"%")
            axPER.set_title(f'XPL% - XPT% {mh} \n {problem} with Q-Learning')
            axPER.set_ylabel("Percentage")
            axPER.set_xlabel("Iteration")
            axPER.legend(loc = 'upper right')
            plt.savefig(f'{dirResultado}/Graficos/Percentage_{mh}_{problem}_Q-Learning.pdf')
            plt.close('all')
            print(f'Grafico de exploracion y explotacion realizado para {mh}, problema: {problem} con Q-Learning')
            
            figPER, axPER = plt.subplots()
            axPER.plot(iteraciones, div, color="b")
            axPER.set_title(f'Diversity {mh} \n {problem} with Q-Learning')
            axPER.set_ylabel("Diversity")
            axPER.set_xlabel("Iteration")
            plt.savefig(f'{dirResultado}/Graficos/Diversity_{mh}_{problem}_Q-Learning.pdf')
            plt.close('all')
            print(f'Grafico dediversidad realizado para {mh}, problema: {problem} con Q-Learning')
        
        os.remove('./Resultados/Transitorio/'+nombreArchivo+'.csv')
    
    print("------------------------------------------------------------------------------------------------------------")
    figPER, axPER = plt.subplots()
    axPER.plot(iteraciones, bestFitnessGWO, color="r", label="GWO")
    axPER.plot(iteraciones, bestFitnessSCA, color="b", label="SCA")
    axPER.plot(iteraciones, bestFitnessPSA, color="g", label="PSA")
    axPER.plot(iteraciones, bestFitnessWOA, color="y", label="WOA")
    axPER.set_title(f'Coverage \n {problem}')
    axPER.set_ylabel("Fitness")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc = 'upper right')
    plt.savefig(f'{dirResultado}/Best/fitness_{problem}_Q-Learning.pdf')
    plt.close('all')
    print(f'Grafico de fitness realizado {problem} ')
    
    figPER, axPER = plt.subplots()
    axPER.plot(iteraciones, bestTimeGWO, color="r", label="GWO")
    axPER.plot(iteraciones, bestTimeSCA, color="b", label="SCA")
    axPER.plot(iteraciones, bestTimePSA, color="g", label="PSA")
    axPER.plot(iteraciones, bestTimeWOA, color="y", label="WOA")
    axPER.set_title(f'Time (s) \n {problem}')
    axPER.set_ylabel("Time (s)")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc = 'upper right')
    plt.savefig(f'{dirResultado}/Best/time_{problem}_Q-Learning.pdf')
    plt.close('all')
    print(f'Grafico de time realizado {problem} ')
    
    figPER, axPER = plt.subplots()
    axPER.plot(iteraciones, bestTfsGWO, color="r", label="GWO")
    axPER.plot(iteraciones, bestTfsSCA, color="b", label="SCA")
    axPER.plot(iteraciones, bestTfsPSA, color="g", label="PSA")
    axPER.plot(iteraciones, bestTfsWOA, color="y", label="WOA")
    axPER.set_title(f'Total feature selected \n {problem}')
    axPER.set_ylabel("Total feature selected")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc = 'upper right')
    plt.savefig(f'{dirResultado}/Best/tfs_{problem}_Q-Learning.pdf')
    plt.close('all')
    print(f'Grafico de total feature selected realizado {problem} ')
    
    figPER, axPER = plt.subplots()
    axPER.plot(iteraciones, bestAccuracyGWO, color="r", label="GWO")
    axPER.plot(iteraciones, bestAccuracySCA, color="b", label="SCA")
    axPER.plot(iteraciones, bestAccuracyPSA, color="g", label="PSA")
    axPER.plot(iteraciones, bestAccuracyWOA, color="y", label="WOA")
    axPER.set_title(f'Accuracy \n {problem}')
    axPER.set_ylabel("Accuracy")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc = 'lower right')
    plt.savefig(f'{dirResultado}/Best/accuracy_{problem}_Q-Learning.pdf')
    plt.close('all')
    print(f'Grafico de accuracy realizado {problem} ')
    
    figPER, axPER = plt.subplots()
    axPER.plot(iteraciones, bestFscoreGWO, color="r", label="GWO")
    axPER.plot(iteraciones, bestFscoreSCA, color="b", label="SCA")
    axPER.plot(iteraciones, bestFscorePSA, color="g", label="PSA")
    axPER.plot(iteraciones, bestFscoreWOA, color="y", label="WOA")
    axPER.set_title(f'f-score \n {problem}')
    axPER.set_ylabel("f-score")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc = 'lower right')
    plt.savefig(f'{dirResultado}/Best/fscore_{problem}_Q-Learning.pdf')
    plt.close('all')
    print(f'Grafico de f-score realizado {problem} ')
    
    figPER, axPER = plt.subplots()
    axPER.plot(iteraciones, bestPrecisionGWO, color="r", label="GWO")
    axPER.plot(iteraciones, bestPrecisionSCA, color="b", label="SCA")
    axPER.plot(iteraciones, bestPrecisionPSA, color="g", label="PSA")
    axPER.plot(iteraciones, bestPrecisionWOA, color="y", label="WOA")
    axPER.set_title(f'Precision \n {problem}')
    axPER.set_ylabel("Precision")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc = 'lower right')
    plt.savefig(f'{dirResultado}/Best/precision_{problem}_Q-Learning.pdf')
    plt.close('all')
    print(f'Grafico de precision realizado {problem} ')
    
    figPER, axPER = plt.subplots()
    axPER.plot(iteraciones, bestRecallGWO, color="r", label="GWO")
    axPER.plot(iteraciones, bestRecallSCA, color="b", label="SCA")
    axPER.plot(iteraciones, bestRecallPSA, color="g", label="PSA")
    axPER.plot(iteraciones, bestRecallWOA, color="y", label="WOA")
    axPER.set_title(f'Recall \n {problem}')
    axPER.set_ylabel("Recall")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc = 'lower right')
    plt.savefig(f'{dirResultado}/Best/recall_{problem}_Q-Learning.pdf')
    plt.close('all')
    print(f'Grafico de recall realizado {problem} ')
    
    figPER, axPER = plt.subplots()
    axPER.plot(iteraciones, bestMCCGWO, color="r", label="GWO")
    axPER.plot(iteraciones, bestMCCSCA, color="b", label="SCA")
    axPER.plot(iteraciones, bestMCCPSA, color="g", label="PSA")
    axPER.plot(iteraciones, bestMCCWOA, color="y", label="WOA")
    axPER.set_title(f'Matthew’s correlation coefficients \n {problem}')
    axPER.set_ylabel("Matthew’s correlation coefficients")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc = 'lower right')
    plt.savefig(f'{dirResultado}/Best/mcc_{problem}_Q-Learning.pdf')
    plt.close('all')
    print(f'Grafico de MCC realizado {problem} ')
    
    figPER, axPER = plt.subplots()
    axPER.plot(iteraciones, bestErrorRateGWO, color="r", label="GWO")
    axPER.plot(iteraciones, bestErrorRateSCA, color="b", label="SCA")
    axPER.plot(iteraciones, bestErrorRatePSA, color="g", label="PSA")
    axPER.plot(iteraciones, bestErrorRateWOA, color="y", label="WOA")
    axPER.set_title(f'Error Rate \n {problem}')
    axPER.set_ylabel("Error Rate")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc = 'upper right')
    plt.savefig(f'{dirResultado}/Best/errorRate_{problem}_Q-Learning.pdf')
    plt.close('all')
    print(f'Grafico de error rate realizado {problem} ')
    
    
    archivoFitness.close()
    archivoAccuracy.close()
    archivoMCC.close()
    print("------------------------------------------------------------------------------------------------------------")
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------
    datos = pd.read_csv(dirResultado+"/fitness_"+problema+'.csv')
    figFitness, axFitness = plt.subplots()
    axFitness = sns.boxplot(x='MH', y='FITNESS', data=datos)
    axFitness.set_title(f'Fitness \n{problema}', loc="center", fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': 'black'})
    axFitness.set_ylabel("Fitness")
    axFitness.set_xlabel("Metaheuristics")
    figFitness.savefig(dirResultado+"/boxplot/boxplot_fitness_"+problema+'_Q-Learning.pdf')
    plt.close('all')
    print(f'Grafico de boxplot con fitness para la instancia {problema} realizado con exito')
    
    figFitness, axFitness = plt.subplots()
    axFitness = sns.violinplot(x='MH', y='FITNESS', data=datos, gridsize=50)
    axFitness.set_title(f'Fitness \n{problema}', loc="center", fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': 'black'})
    axFitness.set_ylabel("Fitness")
    axFitness.set_xlabel("Metaheuristics")
    figFitness.savefig(dirResultado+"/violinplot/violinplot_fitness_"+problema+'_Q-Learning.pdf')
    plt.close('all')
    print(f'Grafico de violines con fitness para la instancia {problema} realizado con exito')
    
    os.remove(dirResultado+"/fitness_"+problema+'.csv')
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------
    datos = pd.read_csv(dirResultado+"/accuracy_"+problema+'.csv')
    figFitness, axFitness = plt.subplots()
    axFitness = sns.boxplot(x='MH', y='ACCURACY', data=datos)
    axFitness.set_title(f'Accuracy \n{problema}', loc="center", fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': 'black'})
    axFitness.set_ylabel("Accuracy")
    axFitness.set_xlabel("Metaheuristics")
    figFitness.savefig(dirResultado+"/boxplot/boxplot_accuracy_"+problema+'_Q-Learning.pdf')
    plt.close('all')
    print(f'Grafico de boxplot con accuracy para la instancia {problema} realizado con exito')
    
    datos = pd.read_csv(dirResultado+"/accuracy_"+problema+'.csv')
    figFitness, axFitness = plt.subplots()
    axFitness = sns.violinplot(x='MH', y='ACCURACY', data=datos, gridsize=50)
    axFitness.set_title(f'Accuracy \n{problema}', loc="center", fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': 'black'})
    axFitness.set_ylabel("Accuracy")
    axFitness.set_xlabel("Metaheuristics")
    figFitness.savefig(dirResultado+"/violinplot/violinplot_accuracy_"+problema+'_Q-Learning.pdf')
    plt.close('all')
    print(f'Grafico de violines con accuracy para la instancia {problema} realizado con exito')
    
    os.remove(dirResultado+"/accuracy_"+problema+'.csv')
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------
    datos = pd.read_csv(dirResultado+"/mcc_"+problema+'.csv')
    figFitness, axFitness = plt.subplots()
    axFitness = sns.boxplot(x='MH', y='MCC', data=datos)
    axFitness.set_title(f'Matthew’s correlation coefficients \n{problema}', loc="center", fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': 'black'})
    axFitness.set_ylabel("Matthew’s correlation coefficients")
    axFitness.set_xlabel("Metaheuristics")
    figFitness.savefig(dirResultado+"/boxplot/boxplot_mcc_"+problema+'_Q-Learning.pdf')
    plt.close('all')
    print(f'Grafico de boxplot con MCC para la instancia {problema} realizado con exito')
    
    datos = pd.read_csv(dirResultado+"/mcc_"+problema+'.csv')
    figFitness, axFitness = plt.subplots()
    axFitness = sns.violinplot(x='MH', y='MCC', data=datos, gridsize=50)
    axFitness.set_title(f'Matthew’s correlation coefficients \n{problema}', loc="center", fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': 'black'})
    axFitness.set_ylabel("Matthew’s correlation coefficients")
    axFitness.set_xlabel("Metaheuristics")
    figFitness.savefig(dirResultado+"/violinplot/violinplot_mcc_"+problema+'_Q-Learning.pdf')
    plt.close('all')
    print(f'Grafico de violines con MCC para la instancia {problema} realizado con exito')
    
    os.remove(dirResultado+"/mcc_"+problema+'.csv')
    
    print("------------------------------------------------------------------------------------------------------------")

    archivoResumenFitness.close()
    archivoResumenPercentage.close()
        