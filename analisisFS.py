
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

from util import util
from BD.sqlite import BD


dirResultado = './Resultados/'
archivoResumenFitness = open(f'{dirResultado}resumen_fitness_FS.csv', 'w')
archivoResumenTimes = open(f'{dirResultado}resumen_times_FS.csv', 'w')
archivoResumenAccuracy = open(f'{dirResultado}resumen_accuracy_FS.csv', 'w')
archivoResumenFScore = open(f'{dirResultado}resumen_fscore_FS.csv', 'w')
archivoResumenPrecision = open(f'{dirResultado}resumen_precision_FS.csv', 'w')
archivoResumenRecall = open(f'{dirResultado}resumen_recall_FS.csv', 'w')
archivoResumenMCC = open(f'{dirResultado}resumen_mcc_FS.csv', 'w')
archivoResumenErrorRate = open(f'{dirResultado}resumen_error_rate_FS.csv', 'w')
archivoResumenTFS = open(f'{dirResultado}resumen_tfs_FS.csv', 'w')
archivoResumenPercentage = open(f'{dirResultado}resumen_percentage_FS.csv', 'w')


archivoResumenFitness.write("instance, best,avg,std-dev, best,avg,std-dev, best,avg,std-dev, best,avg,std-dev, best,avg,std-dev \n")
archivoResumenTimes.write("instance, min time (s),avg. time (s),std-dev time (s), min time (s),avg. time (s),std-dev time (s),min time (s),avg. time (s),std-dev time (s), min time (s),avg. time (s),std-dev time (s), min time (s),avg. time (s),std-dev time (s)\n")
archivoResumenAccuracy.write("instance, best,avg,std-dev, best,avg,std-dev, best,avg,std-dev, best,avg,std-dev, best,avg,std-dev \n")
archivoResumenFScore.write("instance, best,avg,std-dev, best,avg,std-dev, best,avg,std-dev, best,avg,std-dev, best,avg,std-dev \n")
archivoResumenPrecision.write("instance, best,avg,std-dev, best,avg,std-dev, best,avg,std-dev, best,avg,std-dev, best,avg,std-dev \n")
archivoResumenRecall.write("instance, best,avg,std-dev, best,avg,std-dev, best,avg,std-dev, best,avg,std-dev, best,avg,std-dev \n")
archivoResumenMCC.write("instance, best,avg,std-dev, best,avg,std-dev, best,avg,std-dev, best,avg,std-dev, best,avg,std-dev \n")
archivoResumenErrorRate.write("instance, best,avg,std-dev, best,avg,std-dev, best,avg,std-dev, best,avg,std-dev, best,avg,std-dev \n")
archivoResumenTFS.write("instance, avg,std-dev, avg,std-dev, avg,std-dev, avg,std-dev, best,avg,std-dev \n")
archivoResumenPercentage.write("instance, avg. XPL%, avg. XPT%, avg. XPL%, avg. XPT%, avg. XPL%, avg. XPT%, avg. XPL%, avg. XPT%, avg. XPL%, avg. XPT%\n")




graficos = False

bd = BD()

instancias = bd.obtenerInstancias(f'''
                                  "nefrologia","only_clinic"
                                  ''')
print(instancias)
for instancia in instancias:
    
    print(instancia)

    blob = bd.obtenerArchivos(instancia[1])
    corrida = 1

    
    archivoFitness = open(f'{dirResultado}fitness_'+instancia[1]+'.csv', 'w')
    archivoFitness.write('MH,FITNESS\n')
    
    archivoAccuracy = open(f'{dirResultado}accuracy_'+instancia[1]+'.csv', 'w')
    archivoAccuracy.write('MH,ACCURACY\n')
    
    archivoMCC = open(f'{dirResultado}mcc_'+instancia[1]+'.csv', 'w')
    archivoMCC.write('MH,MCC\n')

    fitnessSCA = [] 
    fitnessGWO = [] 
    fitnessWOA = [] 
    fitnessPSA = []
    fitnessMFO = []

    timeSCA = []
    timeGWO = []
    timeWOA = []
    timePSA = []
    timeMFO = []

    xplSCA = [] 
    xplGWO = [] 
    xplWOA = [] 
    xplPSA = []
    xplMFO = []

    xptSCA = []
    xptGWO = []
    xptWOA = []
    xptPSA = []
    xptMFO = []
    
    tfsSCA = []
    tfsGWO = []
    tfsWOA = []
    tfsPSA = []
    tfsMFO = []
    
    accuracySCA = []
    accuracyGWO = []
    accuracyWOA = []
    accuracyPSA = []
    accuracyMFO = []
    
    f1ScoreSCA = []
    f1ScoreGWO = []
    f1ScoreWOA = []
    f1ScorePSA = []
    f1ScoreMFO = []
    
    precisionSCA = []
    precisionGWO = []
    precisionWOA = []
    precisionPSA = []
    precisionMFO = []
    
    recallSCA = []
    recallGWO = []
    recallWOA = []
    recallPSA = []
    recallMFO = []
    
    mccSCA = []
    mccGWO = []
    mccWOA = []
    mccPSA = []
    mccMFO = []
    
    errorRateSCA = []
    errorRateGWO = []
    errorRateWOA = []
    errorRatePSA = []
    errorRateMFO = []
    
   
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

    for d in blob:
        
        nombreArchivo = d[0]
        archivo = d[1]

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
            
            
            archivoFitness.write(f'PSA,{str(np.min(fitness))}\n')
            archivoAccuracy.write(f'PSA,{str(np.average(accuracy))}\n')
            archivoMCC.write(f'PSA,{str(np.average(mcc))}\n')
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
            
            
            archivoFitness.write(f'SCA,{str(np.min(fitness))}\n')
            archivoAccuracy.write(f'SCA,{str(np.average(accuracy))}\n')
            archivoMCC.write(f'SCA,{str(np.average(mcc))}\n')
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
            
            
            archivoFitness.write(f'GWO,{str(np.min(fitness))}\n')
            archivoAccuracy.write(f'GWO,{str(np.average(accuracy))}\n')
            archivoMCC.write(f'GWO,{str(np.average(mcc))}\n')
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
            
            
            archivoFitness.write(f'WOA,{str(np.min(fitness))}\n')
            archivoAccuracy.write(f'WOA,{str(np.average(accuracy))}\n')
            archivoMCC.write(f'WOA,{str(np.average(mcc))}\n')
            
        if mh == 'MFO':
            fitnessMFO.append(fitness[ultimo])
            timeMFO.append(np.round(np.sum(time),3))
            xplMFO.append(np.round(np.mean(xpl), decimals=2))
            xptMFO.append(np.round(np.mean(xpt), decimals=2))
            tfsMFO.append(tfs[ultimo])
            accuracyMFO.append(accuracy[ultimo])
            f1ScoreMFO.append(f1Score[ultimo])
            precisionMFO.append(precision[ultimo])
            recallMFO.append(recall[ultimo])
            mccMFO.append(mcc[ultimo])
            errorRateMFO.append(errorRate[ultimo])
            
            
            archivoFitness.write(f'MFO,{str(np.min(fitness))}\n')
            archivoAccuracy.write(f'MFO,{str(np.average(accuracy))}\n')
            archivoMCC.write(f'MFO,{str(np.average(mcc))}\n')





        # if graficos:







            # fig , ax = plt.subplots()
            # ax.plot(iteraciones,fitness)
            # ax.set_title(f'Convergence {mh} \n {problem} run {corrida}')
            # ax.set_ylabel("Fitness")
            # ax.set_xlabel("Iteration")
            # plt.savefig(f'{dirResultado}/Graficos/Coverange_{mh}_{problem}_{corrida}.pdf')
            # plt.close('all')
            # print(f'Grafico de covergencia realizado {mh} {problem} ')

            # fig , ax = plt.subplots()
            # ax.plot(iteraciones,time, label=r"$\overline{Runtime}$"+": "+str(np.round(np.mean(time), decimals=2))+"(s)\n Runtime: "+str(np.round(np.sum(time),3))+"(s)")
            # ax.set_title(f'Time {mh} \n {problem} run {corrida}')
            # ax.set_ylabel("Time")
            # ax.set_xlabel("Iteration")
            # ax.legend(loc = 'lower right')
            # plt.savefig(f'{dirResultado}/Graficos/Time_{mh}_{problem}_{corrida}.pdf')
            # plt.close('all')
            # print(f'Grafico de tiempo realizado {mh} {problem} ')
            
            # fig , ax = plt.subplots()
            # ax.plot(iteraciones, accuracy)
            # ax.set_title(f'Accuracy {mh} \n {problem} run {corrida}')
            # ax.set_ylabel("Accuracy")
            # ax.set_xlabel("Iteration")
            # plt.savefig(f'{dirResultado}/Graficos/Accuracy_{mh}_{problem}_{corrida}.pdf')
            # plt.close('all')
            # print(f'Grafico de accuracy realizado {mh} {problem} ')

            # fig , ax = plt.subplots()
            # ax.plot(iteraciones,f1Score)
            # ax.set_title(f'F1-Score {mh} \n {problem} run {corrida}')
            # ax.set_ylabel("F1-Score")
            # ax.set_xlabel("Iteration")
            # plt.savefig(f'{dirResultado}/Graficos/F1Score_{mh}_{problem}_{corrida}.pdf')
            # plt.close('all')
            # print(f'Grafico de f1-score realizado {mh} {problem} ')

            # fig , ax = plt.subplots()
            # ax.plot(iteraciones,precision)
            # ax.set_title(f'Precision {mh} \n {problem} run {corrida}')
            # ax.set_ylabel("Precision")
            # ax.set_xlabel("Iteration")
            # plt.savefig(f'{dirResultado}/Graficos/Precision_{mh}_{problem}_{corrida}.pdf')
            # plt.close('all')
            # print(f'Grafico de precision realizado {mh} {problem} ')

            # fig , ax = plt.subplots()
            # ax.plot(iteraciones,recall)
            # ax.set_title(f'Recall {mh} \n {problem} run {corrida}')
            # ax.set_ylabel("Recall")
            # ax.set_xlabel("Iteration")
            # plt.savefig(f'{dirResultado}/Graficos/Recall_{mh}_{problem}_{corrida}.pdf')
            # plt.close('all')
            # print(f'Grafico de recall realizado {mh} {problem} ')

            # fig , ax = plt.subplots()
            # ax.plot(iteraciones,mcc)
            # ax.set_title(f'MCC {mh} \n {problem} run {corrida}')
            # ax.set_ylabel("MCC")
            # ax.set_xlabel("Iteration")
            # plt.savefig(f'{dirResultado}/Graficos/MCC_{mh}_{problem}_{corrida}.pdf')
            # plt.close('all')
            # print(f'Grafico de MCC realizado {mh} {problem} ')

            # fig , ax = plt.subplots()
            # ax.plot(iteraciones,errorRate)
            # ax.set_title(f'Error Rate {mh} \n {problem} run {corrida}')
            # ax.set_ylabel("Error Rate")
            # ax.set_xlabel("Iteration")
            # plt.savefig(f'{dirResultado}/Graficos/ErrorRate_{mh}_{problem}_{corrida}.pdf')
            # plt.close('all')
            # print(f'Grafico de error rate realizado {mh} {problem} ')

            # fig , ax = plt.subplots()
            # ax.plot(iteraciones,tfs)
            # ax.set_title(f'Total Feature Selected {mh} \n {problem} run {corrida}')
            # ax.set_ylabel("Total Feature Selected")
            # ax.set_xlabel("Iteration")
            # plt.savefig(f'{dirResultado}/Graficos/Tfs_{mh}_{problem}_{corrida}.pdf')
            # plt.close('all')
            # print(f'Grafico de total feature selected realizado {mh} {problem} ')





#             figPER, axPER = plt.subplots()
#             axPER.plot(iteraciones, xpl, color="r", label=r"$\overline{XPL}$"+": "+str(np.round(np.mean(xpl), decimals=2))+"%")
#             axPER.plot(iteraciones, xpt, color="b", label=r"$\overline{XPT}$"+": "+str(np.round(np.mean(xpt), decimals=2))+"%")
#             axPER.set_title(f'XPL% - XPT% {mh} \n {problem} run {corrida}')
#             axPER.set_ylabel("Percentage")
#             axPER.set_xlabel("Iteration")
#             axPER.legend(loc = 'upper right')
#             plt.savefig(f'{dirResultado}/Graficos/Percentage_{mh}_{problem}_{corrida}.pdf')
#             plt.close('all')
#             print(f'Grafico de exploracion y explotacion realizado para {mh}, problema: {problem}, corrida: {corrida} ')
        
#         corrida +=1
        
#         if corrida == 32:
#             corrida = 1
        
        os.remove('./Resultados/Transitorio/'+nombreArchivo+'.csv')
    
#     # print(fitnessGWO)
#     # print(fitnessPSA)
#     # print(fitnessSCA)
#     # print(fitnessWOA)
     
    # print(f'''
    #     {problem},{np.min(fitnessPSA)},{np.round(np.average(fitnessPSA),3)},{np.round(np.std(fitnessPSA),3)},{np.min(fitnessSCA)},{np.round(np.average(fitnessSCA),3)},{np.round(np.std(fitnessSCA),3)},{np.min(fitnessMFO)},{np.round(np.average(fitnessMFO),3)},{np.round(np.std(fitnessMFO),3)}
    # ''')
    # print(f'''
    #     {problem},{np.min(timePSA)},{np.round(np.average(timePSA),3)},{np.round(np.std(timePSA),3)},{np.min(timeSCA)},{np.round(np.average(timeSCA),3)},{np.round(np.std(timeSCA),3)},{np.min(timeMFO)},{np.round(np.average(timeMFO),3)},{np.round(np.std(timeMFO),3)}
    # ''')
    # print(f'''
    #     {problem},{np.round(np.average(xplPSA),3)},{np.round(np.average(xptPSA),3)},{np.round(np.average(xplSCA),3)},{np.round(np.average(xptSCA),3)},{np.round(np.average(xplMFO),3)},{np.round(np.average(xplMFO),3)}
    # ''')
    
    # {np.min(fitnessGWO)},{np.round(np.average(fitnessGWO),3)},{np.round(np.std(fitnessGWO),3)}
    # {np.min(fitnessMFO)},{np.round(np.average(fitnessMFO),3)},{np.round(np.std(fitnessMFO),3)}
    # {np.min(fitnessPSA)},{np.round(np.average(fitnessPSA),3)},{np.round(np.std(fitnessPSA),3)}
    # {np.min(fitnessSCA)},{np.round(np.average(fitnessSCA),3)},{np.round(np.std(fitnessSCA),3)}
    # {np.min(fitnessWOA)},{np.round(np.average(fitnessWOA),3)},{np.round(np.std(fitnessWOA),3)}
    archivoResumenFitness.write(f'''{problem},{np.min(fitnessGWO)},{np.round(np.average(fitnessGWO),3)},{np.round(np.std(fitnessGWO),3)},{np.min(fitnessPSA)},{np.round(np.average(fitnessPSA),3)},{np.round(np.std(fitnessPSA),3)},{np.min(fitnessSCA)},{np.round(np.average(fitnessSCA),3)},{np.round(np.std(fitnessSCA),3)},{np.min(fitnessWOA)},{np.round(np.average(fitnessWOA),3)},{np.round(np.std(fitnessWOA),3)} \n''')
    
    
    # {np.min(timeGWO)},{np.round(np.average(timeGWO),3)},{np.round(np.std(timeGWO),3)}
    # {np.min(timeMFO)},{np.round(np.average(timeMFO),3)},{np.round(np.std(timeMFO),3)}
    # {np.min(timePSA)},{np.round(np.average(timePSA),3)},{np.round(np.std(timePSA),3)}
    # {np.min(timeSCA)},{np.round(np.average(timeSCA),3)},{np.round(np.std(timeSCA),3)}
    # {np.min(timeWOA)},{np.round(np.average(timeWOA),3)},{np.round(np.std(timeWOA),3)}    
    archivoResumenTimes.write(f'''{problem},{np.min(timeGWO)},{np.round(np.average(timeGWO),3)},{np.round(np.std(timeGWO),3)},{np.min(timePSA)},{np.round(np.average(timePSA),3)},{np.round(np.std(timePSA),3)},{np.min(timeSCA)},{np.round(np.average(timeSCA),3)},{np.round(np.std(timeSCA),3)},{np.min(timeWOA)},{np.round(np.average(timeWOA),3)},{np.round(np.std(timeWOA),3)} \n''')
    
    
    # {np.max(accuracyGWO)},{np.round(np.average(accuracyGWO),3)},{np.round(np.std(accuracyGWO),3)}
    # {np.max(accuracyMFO)},{np.round(np.average(accuracyMFO),3)},{np.round(np.std(accuracyMFO),3)}
    # {np.max(accuracyPSA)},{np.round(np.average(accuracyPSA),3)},{np.round(np.std(accuracyPSA),3)}
    # {np.max(accuracySCA)},{np.round(np.average(accuracySCA),3)},{np.round(np.std(accuracySCA),3)}
    # {np.max(accuracyWOA)},{np.round(np.average(accuracyWOA),3)},{np.round(np.std(accuracyWOA),3)}
    archivoResumenAccuracy.write(f'''{problem},{np.max(accuracyGWO)},{np.round(np.average(accuracyGWO),3)},{np.round(np.std(accuracyGWO),3)},{np.max(accuracyPSA)},{np.round(np.average(accuracyPSA),3)},{np.round(np.std(accuracyPSA),3)},{np.max(accuracySCA)},{np.round(np.average(accuracySCA),3)},{np.round(np.std(accuracySCA),3)},{np.max(accuracyWOA)},{np.round(np.average(accuracyWOA),3)},{np.round(np.std(accuracyWOA),3)} \n''')
    
    
    # {np.max(f1ScoreGWO)},{np.round(np.average(f1ScoreGWO),3)},{np.round(np.std(f1ScoreGWO),3)}
    # {np.max(f1ScoreMFO)},{np.round(np.average(f1ScoreMFO),3)},{np.round(np.std(f1ScoreMFO),3)}
    # {np.max(f1ScorePSA)},{np.round(np.average(f1ScorePSA),3)},{np.round(np.std(f1ScorePSA),3)}
    # {np.max(f1ScoreSCA)},{np.round(np.average(f1ScoreSCA),3)},{np.round(np.std(f1ScoreSCA),3)}
    # {np.max(f1ScoreWOA)},{np.round(np.average(f1ScoreWOA),3)},{np.round(np.std(f1ScoreWOA),3)}
    archivoResumenFScore.write(f'''{problem},{np.max(f1ScoreGWO)},{np.round(np.average(f1ScoreGWO),3)},{np.round(np.std(f1ScoreGWO),3)},{np.max(f1ScorePSA)},{np.round(np.average(f1ScorePSA),3)},{np.round(np.std(f1ScorePSA),3)},{np.max(f1ScoreSCA)},{np.round(np.average(f1ScoreSCA),3)},{np.round(np.std(f1ScoreSCA),3)},{np.max(f1ScoreWOA)},{np.round(np.average(f1ScoreWOA),3)},{np.round(np.std(f1ScoreWOA),3)} \n''')
    
    
    # {np.max(precisionGWO)},{np.round(np.average(precisionGWO),3)},{np.round(np.std(precisionGWO),3)}
    # {np.max(precisionMFO)},{np.round(np.average(precisionMFO),3)},{np.round(np.std(precisionMFO),3)}
    # {np.max(precisionPSA)},{np.round(np.average(precisionPSA),3)},{np.round(np.std(precisionPSA),3)}
    # {np.max(precisionSCA)},{np.round(np.average(precisionSCA),3)},{np.round(np.std(precisionSCA),3)}
    # {np.max(precisionWOA)},{np.round(np.average(precisionWOA),3)},{np.round(np.std(precisionWOA),3)}
    archivoResumenPrecision.write(f'''{problem},{np.max(precisionGWO)},{np.round(np.average(precisionGWO),3)},{np.round(np.std(precisionGWO),3)},{np.max(precisionPSA)},{np.round(np.average(precisionPSA),3)},{np.round(np.std(precisionPSA),3)},{np.max(precisionSCA)},{np.round(np.average(precisionSCA),3)},{np.round(np.std(precisionSCA),3)},{np.max(precisionWOA)},{np.round(np.average(precisionWOA),3)},{np.round(np.std(precisionWOA),3)} \n''')
    
    
    # {np.max(recallGWO)},{np.round(np.average(recallGWO),3)},{np.round(np.std(recallGWO),3)}
    # {np.max(recallMFO)},{np.round(np.average(recallMFO),3)},{np.round(np.std(recallMFO),3)}
    # {np.max(recallPSA)},{np.round(np.average(recallPSA),3)},{np.round(np.std(recallPSA),3)}
    # {np.max(recallSCA)},{np.round(np.average(recallSCA),3)},{np.round(np.std(recallSCA),3)}
    # {np.max(recallWOA)},{np.round(np.average(recallWOA),3)},{np.round(np.std(recallWOA),3)}
    archivoResumenRecall.write(f'''{problem},{np.max(recallGWO)},{np.round(np.average(recallGWO),3)},{np.round(np.std(recallGWO),3)},{np.max(recallPSA)},{np.round(np.average(recallPSA),3)},{np.round(np.std(recallPSA),3)},{np.max(recallSCA)},{np.round(np.average(recallSCA),3)},{np.round(np.std(recallSCA),3)},{np.max(recallWOA)},{np.round(np.average(recallWOA),3)},{np.round(np.std(recallWOA),3)} \n''')
    
    
    # {np.max(mccGWO)},{np.round(np.average(mccGWO),3)},{np.round(np.std(mccGWO),3)}
    # {np.max(mccMFO)},{np.round(np.average(mccMFO),3)},{np.round(np.std(mccMFO),3)}
    # {np.max(mccPSA)},{np.round(np.average(mccPSA),3)},{np.round(np.std(mccPSA),3)}
    # {np.max(mccSCA)},{np.round(np.average(mccSCA),3)},{np.round(np.std(mccSCA),3)}
    # {np.max(mccWOA)},{np.round(np.average(mccWOA),3)},{np.round(np.std(mccWOA),3)}
    archivoResumenMCC.write(f'''{problem},{np.max(mccGWO)},{np.round(np.average(mccGWO),3)},{np.round(np.std(mccGWO),3)},{np.max(mccPSA)},{np.round(np.average(mccPSA),3)},{np.round(np.std(mccPSA),3)},{np.max(mccSCA)},{np.round(np.average(mccSCA),3)},{np.round(np.std(mccSCA),3)},{np.max(mccWOA)},{np.round(np.average(mccWOA),3)},{np.round(np.std(mccWOA),3)} \n''')
    
    
    # {np.min(errorRateGWO)},{np.round(np.average(errorRateGWO),3)},{np.round(np.std(errorRateGWO),3)}
    # {np.min(errorRateMFO)},{np.round(np.average(errorRateMFO),3)},{np.round(np.std(errorRateMFO),3)}
    # {np.min(errorRatePSA)},{np.round(np.average(errorRatePSA),3)},{np.round(np.std(errorRatePSA),3)}
    # {np.min(errorRateSCA)},{np.round(np.average(errorRateSCA),3)},{np.round(np.std(errorRateSCA),3)}
    # {np.min(errorRateWOA)},{np.round(np.average(errorRateWOA),3)},{np.round(np.std(errorRateWOA),3)}
    archivoResumenErrorRate.write(f'''{problem},{np.min(errorRateGWO)},{np.round(np.average(errorRateGWO),3)},{np.round(np.std(errorRateGWO),3)},{np.min(errorRatePSA)},{np.round(np.average(errorRatePSA),3)},{np.round(np.std(errorRatePSA),3)},{np.min(errorRateSCA)},{np.round(np.average(errorRateSCA),3)},{np.round(np.std(errorRateSCA),3)},{np.min(errorRateWOA)},{np.round(np.average(errorRateWOA),3)},{np.round(np.std(errorRateWOA),3)} \n''')
    
    
    # {np.round(np.average(tfsGWO),3)},{np.round(np.std(tfsGWO),3)}
    # {np.round(np.average(tfsMFO),3)},{np.round(np.std(tfsMFO),3)}
    # {np.round(np.average(tfsPSA),3)},{np.round(np.std(tfsPSA),3)}
    # {np.round(np.average(tfsSCA),3)},{np.round(np.std(tfsSCA),3)}
    # {np.round(np.average(tfsWOA),3)},{np.round(np.std(tfsWOA),3)}
    archivoResumenTFS.write(f'''{problem},{np.round(np.average(tfsGWO),3)},{np.round(np.std(tfsGWO),3)},{np.round(np.average(tfsPSA),3)},{np.round(np.std(tfsPSA),3)},{np.round(np.average(tfsSCA),3)},{np.round(np.std(tfsSCA),3)},{np.round(np.average(tfsWOA),3)},{np.round(np.std(tfsWOA),3)} \n''')
    
    
    # {np.round(np.average(xplGWO),3)},{np.round(np.average(xptGWO),3)}
    # {np.round(np.average(xplMFO),3)},{np.round(np.average(xptMFO),3)}
    # {np.round(np.average(xplPSA),3)},{np.round(np.average(xptPSA),3)}
    # {np.round(np.average(xplSCA),3)},{np.round(np.average(xptSCA),3)}
    # {np.round(np.average(xplWOA),3)},{np.round(np.average(xptWOA),3)}
    archivoResumenPercentage.write(f'''{problem},{np.round(np.average(xplGWO),3)},{np.round(np.average(xptGWO),3)},{np.round(np.average(xplPSA),3)},{np.round(np.average(xptPSA),3)},{np.round(np.average(xplSCA),3)},{np.round(np.average(xptSCA),3)},{np.round(np.average(xplWOA),3)},{np.round(np.average(xptWOA),3)} \n''')
    

    
    
    blob = bd.obtenerMejoresArchivos(instancia[1],"")

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
        ax.set_title(f'Diversity {mh} - {problem}')
        ax.set_ylabel("Diversity")
        ax.set_xlabel("Iteration")
        plt.savefig(f'{dirResultado}/Graficos/Diversity_{mh}_{problem}.pdf')
        plt.close('all')
        print(f'Grafico de diversidad realizado {mh} {problem} ')
        
        figPER, axPER = plt.subplots()
        axPER.plot(iteraciones, xpl, color="r", label=r"$\overline{XPL}$"+": "+str(np.round(np.mean(xpl), decimals=2))+"%")
        axPER.plot(iteraciones, xpt, color="b", label=r"$\overline{XPT}$"+": "+str(np.round(np.mean(xpt), decimals=2))+"%")
        axPER.set_title(f'XPL% - XPT% {mh} \n {problem}')
        axPER.set_ylabel("Percentage")
        axPER.set_xlabel("Iteration")
        axPER.legend(loc = 'upper right')
        plt.savefig(f'{dirResultado}/Graficos/Percentage_{mh}_{problem}.pdf')
        plt.close('all')
        print(f'Grafico de exploracion y explotacion realizado para {mh}, problema: {problem}')
        
        
        os.remove('./Resultados/Transitorio/'+nombreArchivo+'.csv')
    
    print("------------------------------------------------------------------------------------------------------------")
    figPER, axPER = plt.subplots()
    axPER.plot(iteraciones, bestFitnessGWO, color="r", label="GWO")
    axPER.plot(iteraciones, bestFitnessSCA, color="b", label="SCA")
    axPER.plot(iteraciones, bestFitnessPSA, color="g", label="PSA")
    axPER.plot(iteraciones, bestFitnessWOA, color="y", label="WOA")
    # axPER.plot(iteraciones, bestFitnessMFO, color="m", label="MFO")
    axPER.set_title(f'Coverage \n {problem}')
    axPER.set_ylabel("Fitness")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc = 'upper right')
    plt.savefig(f'{dirResultado}/Best/fitness_{problem}.pdf')
    plt.close('all')
    print(f'Grafico de fitness realizado {problem} ')
    
    figPER, axPER = plt.subplots()
    axPER.plot(iteraciones, bestTimeGWO, color="r", label="GWO")
    axPER.plot(iteraciones, bestTimeSCA, color="b", label="SCA")
    axPER.plot(iteraciones, bestTimePSA, color="g", label="PSA")
    axPER.plot(iteraciones, bestTimeWOA, color="y", label="WOA")
    # axPER.plot(iteraciones, bestTimeMFO, color="m", label="MFO")
    axPER.set_title(f'Time (s) \n {problem}')
    axPER.set_ylabel("Time (s)")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc = 'upper right')
    plt.savefig(f'{dirResultado}/Best/time_{problem}.pdf')
    plt.close('all')
    print(f'Grafico de time realizado {problem} ')
    
    figPER, axPER = plt.subplots()
    axPER.plot(iteraciones, bestTfsGWO, color="r", label="GWO")
    axPER.plot(iteraciones, bestTfsSCA, color="b", label="SCA")
    axPER.plot(iteraciones, bestTfsPSA, color="g", label="PSA")
    axPER.plot(iteraciones, bestTfsWOA, color="y", label="WOA")
    # axPER.plot(iteraciones, bestTfsMFO, color="m", label="MFO")
    axPER.set_title(f'Total feature selected \n {problem}')
    axPER.set_ylabel("Total feature selected")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc = 'upper right')
    plt.savefig(f'{dirResultado}/Best/tfs_{problem}.pdf')
    plt.close('all')
    print(f'Grafico de total feature selected realizado {problem} ')
    
    figPER, axPER = plt.subplots()
    axPER.plot(iteraciones, bestAccuracyGWO, color="r", label="GWO")
    axPER.plot(iteraciones, bestAccuracySCA, color="b", label="SCA")
    axPER.plot(iteraciones, bestAccuracyPSA, color="g", label="PSA")
    axPER.plot(iteraciones, bestAccuracyWOA, color="y", label="WOA")
    # axPER.plot(iteraciones, bestAccuracyMFO, color="m", label="MFO")
    axPER.set_title(f'Accuracy \n {problem}')
    axPER.set_ylabel("Accuracy")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc = 'lower right')
    plt.savefig(f'{dirResultado}/Best/accuracy_{problem}.pdf')
    plt.close('all')
    print(f'Grafico de accuracy realizado {problem} ')
    
    figPER, axPER = plt.subplots()
    axPER.plot(iteraciones, bestFscoreGWO, color="r", label="GWO")
    axPER.plot(iteraciones, bestFscoreSCA, color="b", label="SCA")
    axPER.plot(iteraciones, bestFscorePSA, color="g", label="PSA")
    axPER.plot(iteraciones, bestFscoreWOA, color="y", label="WOA")
    # axPER.plot(iteraciones, bestFscoreMFO, color="m", label="MFO")
    axPER.set_title(f'f-score \n {problem}')
    axPER.set_ylabel("f-score")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc = 'lower right')
    plt.savefig(f'{dirResultado}/Best/fscore_{problem}.pdf')
    plt.close('all')
    print(f'Grafico de f-score realizado {problem} ')
    
    figPER, axPER = plt.subplots()
    axPER.plot(iteraciones, bestPrecisionGWO, color="r", label="GWO")
    axPER.plot(iteraciones, bestPrecisionSCA, color="b", label="SCA")
    axPER.plot(iteraciones, bestPrecisionPSA, color="g", label="PSA")
    axPER.plot(iteraciones, bestPrecisionWOA, color="y", label="WOA")
    # axPER.plot(iteraciones, bestPrecisionMFO, color="m", label="MFO")
    axPER.set_title(f'Precision \n {problem}')
    axPER.set_ylabel("Precision")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc = 'lower right')
    plt.savefig(f'{dirResultado}/Best/precision_{problem}.pdf')
    plt.close('all')
    print(f'Grafico de precision realizado {problem} ')
    
    figPER, axPER = plt.subplots()
    axPER.plot(iteraciones, bestRecallGWO, color="r", label="GWO")
    axPER.plot(iteraciones, bestRecallSCA, color="b", label="SCA")
    axPER.plot(iteraciones, bestRecallPSA, color="g", label="PSA")
    axPER.plot(iteraciones, bestRecallWOA, color="y", label="WOA")
    # axPER.plot(iteraciones, bestRecallMFO, color="m", label="MFO")
    axPER.set_title(f'Recall \n {problem}')
    axPER.set_ylabel("Recall")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc = 'lower right')
    plt.savefig(f'{dirResultado}/Best/recall_{problem}.pdf')
    plt.close('all')
    print(f'Grafico de recall realizado {problem} ')
    
    figPER, axPER = plt.subplots()
    axPER.plot(iteraciones, bestMCCGWO, color="r", label="GWO")
    axPER.plot(iteraciones, bestMCCSCA, color="b", label="SCA")
    axPER.plot(iteraciones, bestMCCPSA, color="g", label="PSA")
    axPER.plot(iteraciones, bestMCCWOA, color="y", label="WOA")
    # axPER.plot(iteraciones, bestMCCMFO, color="m", label="MFO")
    axPER.set_title(f'Matthew’s correlation coefficients \n {problem}')
    axPER.set_ylabel("Matthew’s correlation coefficients")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc = 'lower right')
    plt.savefig(f'{dirResultado}/Best/mcc_{problem}.pdf')
    plt.close('all')
    print(f'Grafico de MCC realizado {problem} ')
    
    figPER, axPER = plt.subplots()
    axPER.plot(iteraciones, bestErrorRateGWO, color="r", label="GWO")
    axPER.plot(iteraciones, bestErrorRateSCA, color="b", label="SCA")
    axPER.plot(iteraciones, bestErrorRatePSA, color="g", label="PSA")
    axPER.plot(iteraciones, bestErrorRateWOA, color="y", label="WOA")
    # axPER.plot(iteraciones, bestErrorRateMFO, color="m", label="MFO")
    axPER.set_title(f'Error Rate \n {problem}')
    axPER.set_ylabel("Error Rate")
    axPER.set_xlabel("Iteration")
    axPER.legend(loc = 'upper right')
    plt.savefig(f'{dirResultado}/Best/errorRate_{problem}.pdf')
    plt.close('all')
    print(f'Grafico de error rate realizado {problem} ')
    
    
    
    
    
    
    archivoFitness.close()
    archivoAccuracy.close()
    archivoMCC.close()
    print("------------------------------------------------------------------------------------------------------------")
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------
    datos = pd.read_csv(dirResultado+"/fitness_"+instancia[1]+'.csv')
    figFitness, axFitness = plt.subplots()
    axFitness = sns.boxplot(x='MH', y='FITNESS', data=datos)
    axFitness.set_title(f'Fitness \n{instancia[1]}', loc="center", fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': 'black'})
    axFitness.set_ylabel("Fitness")
    axFitness.set_xlabel("Metaheuristics")
    figFitness.savefig(dirResultado+"/boxplot/boxplot_fitness_"+instancia[1]+'.pdf')
    plt.close('all')
    print(f'Grafico de boxplot con fitness para la instancia {instancia[1]} realizado con exito')
    
    figFitness, axFitness = plt.subplots()
    axFitness = sns.violinplot(x='MH', y='FITNESS', data=datos, gridsize=50)
    axFitness.set_title(f'Fitness \n{instancia[1]}', loc="center", fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': 'black'})
    axFitness.set_ylabel("Fitness")
    axFitness.set_xlabel("Metaheuristics")
    figFitness.savefig(dirResultado+"/violinplot/violinplot_fitness_"+instancia[1]+'.pdf')
    plt.close('all')
    print(f'Grafico de violines con fitness para la instancia {instancia[1]} realizado con exito')
    
    os.remove(dirResultado+"/fitness_"+instancia[1]+'.csv')
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------
    datos = pd.read_csv(dirResultado+"/accuracy_"+instancia[1]+'.csv')
    figFitness, axFitness = plt.subplots()
    axFitness = sns.boxplot(x='MH', y='ACCURACY', data=datos)
    axFitness.set_title(f'Accuracy \n{instancia[1]}', loc="center", fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': 'black'})
    axFitness.set_ylabel("Accuracy")
    axFitness.set_xlabel("Metaheuristics")
    figFitness.savefig(dirResultado+"/boxplot/boxplot_accuracy_"+instancia[1]+'.pdf')
    plt.close('all')
    print(f'Grafico de boxplot con accuracy para la instancia {instancia[1]} realizado con exito')
    
    datos = pd.read_csv(dirResultado+"/accuracy_"+instancia[1]+'.csv')
    figFitness, axFitness = plt.subplots()
    axFitness = sns.violinplot(x='MH', y='ACCURACY', data=datos, gridsize=50)
    axFitness.set_title(f'Accuracy \n{instancia[1]}', loc="center", fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': 'black'})
    axFitness.set_ylabel("Accuracy")
    axFitness.set_xlabel("Metaheuristics")
    figFitness.savefig(dirResultado+"/violinplot/violinplot_accuracy_"+instancia[1]+'.pdf')
    plt.close('all')
    print(f'Grafico de violines con accuracy para la instancia {instancia[1]} realizado con exito')
    
    os.remove(dirResultado+"/accuracy_"+instancia[1]+'.csv')
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------
    datos = pd.read_csv(dirResultado+"/mcc_"+instancia[1]+'.csv')
    figFitness, axFitness = plt.subplots()
    axFitness = sns.boxplot(x='MH', y='MCC', data=datos)
    axFitness.set_title(f'Matthew’s correlation coefficients \n{instancia[1]}', loc="center", fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': 'black'})
    axFitness.set_ylabel("Matthew’s correlation coefficients")
    axFitness.set_xlabel("Metaheuristics")
    figFitness.savefig(dirResultado+"/boxplot/boxplot_mcc_"+instancia[1]+'.pdf')
    plt.close('all')
    print(f'Grafico de boxplot con MCC para la instancia {instancia[1]} realizado con exito')
    
    datos = pd.read_csv(dirResultado+"/mcc_"+instancia[1]+'.csv')
    figFitness, axFitness = plt.subplots()
    axFitness = sns.violinplot(x='MH', y='MCC', data=datos, gridsize=50)
    axFitness.set_title(f'Matthew’s correlation coefficients \n{instancia[1]}', loc="center", fontdict={'fontsize': 10, 'fontweight': 'bold', 'color': 'black'})
    axFitness.set_ylabel("Matthew’s correlation coefficients")
    axFitness.set_xlabel("Metaheuristics")
    figFitness.savefig(dirResultado+"/violinplot/violinplot_mcc_"+instancia[1]+'.pdf')
    plt.close('all')
    print(f'Grafico de violines con MCC para la instancia {instancia[1]} realizado con exito')
    
    os.remove(dirResultado+"/mcc_"+instancia[1]+'.csv')
    
    print("------------------------------------------------------------------------------------------------------------")

archivoResumenFitness.close()
archivoResumenTimes.close()
archivoResumenAccuracy.close()
archivoResumenFScore.close()
archivoResumenPrecision.close()
archivoResumenRecall.close()
archivoResumenMCC.close()
archivoResumenErrorRate.close()
archivoResumenTFS.close()
archivoResumenPercentage.close()
        