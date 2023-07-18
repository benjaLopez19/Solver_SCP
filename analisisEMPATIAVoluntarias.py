import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

from util import util
from BD.sqlite import BD

bd = BD()

# instancias = bd.obtenerInstancias(f'''
#                                   "Divorce","breast-cancer-wisconsin","ionosphere","sonar","wdbc"
#                                   ''')
instancias = bd.obtenerInstancias(f'''
                                  'EMPATIA-V01','EMPATIA-V02','EMPATIA-V03','EMPATIA-V04','EMPATIA-V05','EMPATIA-V06','EMPATIA-V07','EMPATIA-V08','EMPATIA-V09','EMPATIA-V10','EMPATIA-V11','EMPATIA-V12','EMPATIA-V13','EMPATIA-V14','EMPATIA-V15','EMPATIA-V16','EMPATIA-V17','EMPATIA-V18','EMPATIA-V19','EMPATIA-V20','EMPATIA-V21','EMPATIA-V22','EMPATIA-V23','EMPATIA-V24','EMPATIA-V25','EMPATIA-V26','EMPATIA-V27','EMPATIA-V28','EMPATIA-V29','EMPATIA-V30','EMPATIA-V31','EMPATIA-V32','EMPATIA-V33','EMPATIA-V34','EMPATIA-V35','EMPATIA-V36','EMPATIA-V37','EMPATIA-V38','EMPATIA-V39','EMPATIA-V40','EMPATIA-V41','EMPATIA-V42'
                                  ''')
# clasificadores = ["KNN","RandomForest","Xgboost"]
clasificadores = [""]

# binarizaciones = ["100","200","300","400","500","600","700","800","900","1000"]

# binarizaciones = ["S4-STD","V4-STD","X4-STD","Z4-STD","S4-COM","V4-COM","X4-COM","Z4-COM","S4-ELIT","V4-ELIT","X4-ELIT","Z4-ELIT"]

dirResultado = './Resultados/'

# --------------------------------------------------------------------------------------- #

mhs = ['WOA','GA']
archivoResumen = open(f'{dirResultado}resumen.csv', 'w')
archivoResumen.write(f''' , f-score, , , , , , times in second, , , , , , TFS, , , \n''')
archivoResumen.write(f''' , WOA, , , GA, , , WOA, , , GA, , , WOA, , GA, \n''')
archivoResumenFitness = open(f'{dirResultado}resumen_fitness.csv', 'w')
archivoResumenFscore = open(f'{dirResultado}resumen_fscore.csv', 'w')
archivoResumenTimes = open(f'{dirResultado}resumen_times.csv', 'w')
archivoResumenTotalFeaturesSelected = open(f'{dirResultado}resumen_total_feature_selected.csv', 'w')
for mh in mhs:
    # archivoResumen.write(f''' ,{mh}, , ,{mh},''')
    archivoResumenFitness.write(f''' ,{mh}, ,''')
    archivoResumenFscore.write(f''' ,{mh}, ,''')
    archivoResumenTimes.write(f''' ,{mh}, ,''')
    archivoResumenTotalFeaturesSelected.write(f''' ,{mh},''')
    
archivoResumenFitness.write(f'''\n''')
archivoResumenFscore.write(f'''\n''')
archivoResumenTimes.write(f'''\n''')
archivoResumenTotalFeaturesSelected.write(f'''\n''')

archivoResumen.write(f'Problema,best,avg,dev-std,best,avg,dev-std,best,avg,dev-std,best,avg,dev-std,avg,dev-std,avg,dev-std\n')
archivoResumenFitness.write(f'Problema,best,avg,dev-std,best,avg,dev-std\n')
archivoResumenFscore.write(f'Problema,best,avg,dev-std,best,avg,dev-std\n')
archivoResumenTimes.write(f'Problema,best,avg,dev-std,best,avg,dev-std\n')
archivoResumenTotalFeaturesSelected.write(f'Problema,avg,dev-std,avg,dev-std\n')

for instancia in instancias:
    print(instancia)
    bss = 500
    clasificador = ''
    blob = bd.obtenerArchivosBSSClasificador(instancia[1],"",bss, clasificador)
    
    fitnessWOA = []
    fscoreWOA = []
    timesWOA = []
    totalFeatureSelectedWOA = []
    
    fitnessGA = []
    fscoreGA = []
    timesGA = []
    totalFeatureSelectedGA = []
    
    for d in blob:
                
        nombreArchivo = d[6]
        archivo = d[7]
        mh = d[1]
        problema = d[5]
        
        direccionDestiono = './Resultados/Transitorio/'+nombreArchivo+'.csv'
                
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
        
        ultimo = len(iteraciones) - 1
        
        
        if mh == 'WOA':
            fitnessWOA.append(fitness[ultimo])
            fscoreWOA.append(f1Score[ultimo])
            timesWOA.append(np.round(np.sum(time),1))
            totalFeatureSelectedWOA.append(tfs[ultimo])
            
        if mh == 'GA':
            fitnessGA.append(fitness[ultimo])
            fscoreGA.append(f1Score[ultimo])
            timesGA.append(np.round(np.sum(time),1))
            totalFeatureSelectedGA.append(tfs[ultimo])
            
        os.remove('./Resultados/Transitorio/'+nombreArchivo+'.csv')
        
    archivoResumenFitness.write(F'''{problema},{np.min(fitnessWOA)},{np.round(np.average(fitnessWOA),3)},{np.round(np.std(fitnessWOA),3)},{np.min(fitnessGA)},{np.round(np.average(fitnessGA),3)},{np.round(np.std(fitnessGA),3)}\n''')
    archivoResumenFscore.write(F'''{problema},{np.max(fscoreWOA)},{np.round(np.average(fscoreWOA),3)},{np.round(np.std(fscoreWOA),3)},{np.max(fscoreGA)},{np.round(np.average(fscoreGA),3)},{np.round(np.std(fscoreGA),3)}\n''')
    archivoResumenTimes.write(F'''{problema},{np.min(timesWOA)},{np.round(np.average(timesWOA),3)},{np.round(np.std(timesWOA),3)},{np.min(timesGA)},{np.round(np.average(timesGA),3)},{np.round(np.std(timesGA),3)}\n''')
    archivoResumenTotalFeaturesSelected.write(F'''{problema},{np.round(np.average(totalFeatureSelectedWOA),1)},{np.round(np.std(totalFeatureSelectedWOA),3)},{np.round(np.average(totalFeatureSelectedGA),1)},{np.round(np.std(totalFeatureSelectedGA),3)}\n''')
    
    archivoResumen.write(F'''{problema},{np.max(fscoreWOA)},{np.round(np.average(fscoreWOA),3)},{np.round(np.std(fscoreWOA),3)},{np.max(fscoreGA)},{np.round(np.average(fscoreGA),3)},{np.round(np.std(fscoreGA),3)},{np.min(timesWOA)},{np.round(np.average(timesWOA),3)},{np.round(np.std(timesWOA),3)},{np.min(timesGA)},{np.round(np.average(timesGA),3)},{np.round(np.std(timesGA),3)},{np.round(np.average(totalFeatureSelectedWOA),1)},{np.round(np.std(totalFeatureSelectedWOA),3)},{np.round(np.average(totalFeatureSelectedGA),1)},{np.round(np.std(totalFeatureSelectedGA),3)}\n''')
# for clasificador in clasificadores:
    
#     archivoResumenFitness = open(f'{dirResultado}resumen_fitness_{clasificador}_WOA.csv', 'w')
#     archivoResumenFscore = open(f'{dirResultado}resumen_fscore_{clasificador}_WOA.csv', 'w')
#     archivoResumenTimes = open(f'{dirResultado}resumen_times_{clasificador}_WOA.csv', 'w')
#     archivoResumenTotalFeaturesSelected = open(f'{dirResultado}resumen_total_feature_selected_{clasificador}_WOA.csv', 'w')
    
#     # for bss in binarizaciones:
#     for instancia in instancias:
#         # archivoResumenFitness.write(f''' ,{bss}, ,''')
#         # # archivoResumenFscore.write(f''' ,{bss}, ,''')
#         # archivoResumenTimes.write(f''' ,{bss}, ,''')
        
#         archivoResumenFitness.write(f''' ,{instancia[1]}, ,''')
#         archivoResumenFscore.write(f''' ,{instancia[1]}, ,''')
#         archivoResumenTimes.write(f''' ,{instancia[1]}, ,''')
#         archivoResumenTotalFeaturesSelected.write(f''' ,{instancia[1]},''')
    
#     # archivoResumenFitness.write(' \n')
#     # archivoResumenFitness.write('instance')
    
#     archivoResumenFitness.write(' \n')
#     archivoResumenFitness.write('BSS')
    
#     archivoResumenFscore.write(' \n')
#     archivoResumenFscore.write('BSS')
    
#     # archivoResumenTimes.write(' \n')
#     # archivoResumenTimes.write('instance')
    
#     archivoResumenTimes.write(' \n')
#     archivoResumenTimes.write('BSS')
    
#     archivoResumenTotalFeaturesSelected.write(' \n')
#     archivoResumenTotalFeaturesSelected.write('BSS')
    
#     # for bss in binarizaciones:
#     for instancia in instancias:
#         archivoResumenFitness.write(f',best,avg,dev-std')
#         archivoResumenFscore.write(f',best,avg,dev-std')
#         archivoResumenTimes.write(f',min,avg,dev-std')
#         archivoResumenTotalFeaturesSelected.write(f',avg,dev-std')
    
#     archivoResumenFitness.write('\n')
#     archivoResumenFscore.write('\n')
#     archivoResumenTimes.write('\n')
#     archivoResumenTotalFeaturesSelected.write('\n')
    
#     primero = True
    
#     # for instancia in instancias:
#     for bss in binarizaciones:
        
#         if primero:
#             # archivoResumenFitness.write(instancia[1])
#             # # archivoResumenFscore.write(instancia[1])
#             # archivoResumenTimes.write(instancia[1])
            
#             archivoResumenFitness.write(bss)
#             archivoResumenFscore.write(bss)
#             archivoResumenTimes.write(bss)
#             archivoResumenTotalFeaturesSelected.write(bss)
#             primero = False
#         else:
#             # archivoResumenFitness.write('\n')
#             # archivoResumenFitness.write(instancia[1])
#             # # archivoResumenFscore.write('\n')
#             # # archivoResumenFscore.write(instancia[1])
#             # archivoResumenTimes.write('\n')
#             # archivoResumenTimes.write(instancia[1])
            
#             archivoResumenFitness.write('\n')
#             archivoResumenFitness.write(bss)
#             archivoResumenFscore.write('\n')
#             archivoResumenFscore.write(bss)
#             archivoResumenTimes.write('\n')
#             archivoResumenTimes.write(bss)
#             archivoResumenTotalFeaturesSelected.write('\n')
#             archivoResumenTotalFeaturesSelected.write(bss)
            
#         for instancia in instancias:
            
#             print("-------------------------------------------------------------------------------------------------------")
#             print(f'''{clasificador} - {instancia[1]} - {bss}''')
            
#             blob = bd.obtenerArchivosBSSClasificador(instancia[1],"",bss, clasificador)
            
#             fitnessPSA = []
#             fscorePSA = []
#             timesPSA = []
#             totalFeatureSelectedPSA = []
            
#             for d in blob:
                
#                 nombreArchivo = d[6]
#                 archivo = d[7]
#                 mh = d[1]
#                 print(mh)
#                 problema = d[5]
                
#                 direccionDestiono = './Resultados/Transitorio/'+nombreArchivo+"_"+clasificador+'.csv'
                
#                 util.writeTofile(archivo,direccionDestiono)
            
#                 data = pd.read_csv(direccionDestiono)
                
                
#                 iteraciones = data['iter']
#                 fitness     = data['fitness']
#                 time        = data['time']
#                 accuracy    = data['accuracy']
#                 f1Score     = data['f1-score']
#                 precision   = data['precision']
#                 recall      = data['recall']
#                 mcc         = data['mcc']
#                 errorRate   = data['errorRate']
#                 tfs         = data['TFS']
#                 xpl         = data['XPL']
#                 xpt         = data['XPT']
#                 div         = data['DIV']
                
#                 ultimo = len(iteraciones) - 1
                
                
#                 if mh == 'WOA':
#                     fitnessPSA.append(fitness[ultimo])
#                     fscorePSA.append(f1Score[ultimo])
#                     timesPSA.append(np.round(np.sum(time),1))
#                     totalFeatureSelectedPSA.append(tfs[ultimo])
                    
#                 os.remove('./Resultados/Transitorio/'+nombreArchivo+"_"+clasificador+'.csv')
                
#             archivoResumenFitness.write(F''',{np.min(fitnessPSA)},{np.round(np.average(fitnessPSA),3)},{np.round(np.std(fitnessPSA),3)}''')
#             archivoResumenFscore.write(F''',{np.max(fscorePSA)},{np.round(np.average(fscorePSA),3)},{np.round(np.std(fscorePSA),3)}''')
#             archivoResumenTimes.write(F''',{np.min(timesPSA)},{np.round(np.average(timesPSA),1)},{np.round(np.std(timesPSA),1)}''')
#             archivoResumenTotalFeaturesSelected.write(F''',{np.round(np.average(totalFeatureSelectedPSA),1)},{np.round(np.std(totalFeatureSelectedPSA),1)}''')
            
            
#             blob = bd.obtenerMejoresArchivosconClasificadorBSS(instancia[1],"",clasificador,bss)
            
#             bestFitnessPSA      = fitness
#             bestTimePSA         = time
#             bestFscorePSA       = f1Score
#             bestDIVPSA          = div 
#             bestXPLPSA          = xpl
#             bestXPTPSA          = xpt
#             bestfsPSA           = tfs
            
            
#             for d in blob:
                
#                 nombreArchivo = d[6]
#                 archivo = d[7]
#                 mh = d[1]
#                 problema = d[5]
                
#                 direccionDestiono = './Resultados/Transitorio/'+nombreArchivo+"_"+clasificador+'.csv'
                
#                 print(nombreArchivo)
                
#                 util.writeTofile(archivo,direccionDestiono)
            
#                 data = pd.read_csv(direccionDestiono)
                
                
#                 iteraciones = data['iter']
#                 fitness     = data['fitness']
#                 time        = data['time']
#                 accuracy    = data['accuracy']
#                 f1Score     = data['f1-score']
#                 precision   = data['precision']
#                 recall      = data['recall']
#                 mcc         = data['mcc']
#                 errorRate   = data['errorRate']
#                 tfs         = data['TFS']
#                 xpl         = data['XPL']
#                 xpt         = data['XPT']
#                 div         = data['DIV']
                
#                 if mh == 'WOA':
#                     bestFitnessPSA      = fitness
#                     bestTimePSA         = time
#                     bestFscorePSA       = f1Score
#                     bestDIVPSA          = div 
#                     bestXPLPSA          = xpl
#                     bestXPTPSA          = xpt
#                     bestfsPSA           = tfs
                
#                     # print("------------------------------------------------------------------------------------------------------------")
#                     figPER, axPER = plt.subplots()
#                     axPER.plot(iteraciones, bestFitnessPSA, color="g", label="WOA_"+bss+": "+str(np.min(bestFitnessPSA)))
#                     # axPER.set_title(f'Coverage - {problema} - {clasificador} - {bss}')
#                     axPER.set_title(f'Coverage - {problema} - {bss}')
#                     axPER.set_ylabel("Fitness")
#                     axPER.set_xlabel("Iteration")
#                     axPER.legend(loc = 'upper right')
#                     plt.savefig(f'{dirResultado}/Best/fitness_{problema}_{clasificador}_{bss}.pdf')
#                     plt.close('all')
#                     print(f'Grafico de fitness realizado - {problema} - {clasificador} - {bss}')
                    
#                     # figPER, axPER = plt.subplots()
#                     # axPER.plot(iteraciones, bestTimePSA, color="g", label="PSA")
#                     # axPER.set_title(f'Time (s) - {problema} - {clasificador} - {bss}')
#                     # axPER.set_ylabel("Time (s)")
#                     # axPER.set_xlabel("Iteration")
#                     # axPER.legend(loc = 'upper right')
#                     # plt.savefig(f'{dirResultado}/Best/time_{problema}_{clasificador}_{bss}.pdf')
#                     # plt.close('all')
#                     # print(f'Grafico de time realizado - {problema} - {clasificador} - {bss}')
                    
#                     figPER, axPER = plt.subplots()
#                     axPER.plot(iteraciones, bestFscorePSA, color="g", label="WOA_"+bss+": "+str(np.max(bestFscorePSA)))
#                     # axPER.set_title(f'f-score - {problema} - {clasificador} - {bss}')
#                     axPER.set_title(f'f-score - {problema} - {bss}')
#                     axPER.set_ylabel("f-score")
#                     axPER.set_xlabel("Iteration")
#                     axPER.legend(loc = 'lower right')
#                     plt.savefig(f'{dirResultado}/Best/fscore_{problema}_{clasificador}_{bss}.pdf')
#                     plt.close('all')
#                     print(f'Grafico de f-score realizado - {problema} - {clasificador} - {bss}')
                    
#                     # fig , ax = plt.subplots()
#                     # ax.plot(iteraciones,bestDIVPSA)
#                     # ax.set_title(f'Diversity {mh} - {problema} - {clasificador} - {bss}')
#                     # ax.set_ylabel("Diversity")
#                     # ax.set_xlabel("Iteration")
#                     # plt.savefig(f'{dirResultado}/Graficos/Diversity_{mh}_{problema}_{clasificador}_{bss}.pdf')
#                     # plt.close('all')
#                     # print(f'Grafico de diversidad realizado - {problema} - {clasificador} - {bss}')
                    
#                     fig , ax = plt.subplots()
#                     # ax.plot(iteraciones,bestfsPSA, color= "g", label="GA_"+bss)
#                     ax.plot(iteraciones, bestfsPSA, color="g", label="WOA_"+bss+": "+str(bestfsPSA[len(iteraciones) - 1]))
#                     # ax.set_title(f'Total Feature Selected {mh} - {problema} - {clasificador} - {bss}')
#                     ax.set_title(f'Total Feature Selected {mh} - {problema} - {bss}')
#                     ax.set_ylabel("Total Feature Selected")
#                     ax.set_xlabel("Iteration")
#                     ax.legend(loc = 'upper right')
#                     plt.savefig(f'{dirResultado}/Graficos/TFS_{mh}_{problema}_{clasificador}_{bss}.pdf')
#                     plt.close('all')
#                     print(f'Grafico de diversidad realizado - {problema} - {clasificador} - {bss}')
                    
#                     # figPER, axPER = plt.subplots()
#                     # axPER.plot(iteraciones, bestXPLPSA, color="r", label=r"$\overline{XPL}$"+": "+str(np.round(np.mean(bestXPLPSA), decimals=2))+"%")
#                     # axPER.plot(iteraciones, bestXPTPSA, color="b", label=r"$\overline{XPT}$"+": "+str(np.round(np.mean(bestXPTPSA), decimals=2))+"%")
#                     # axPER.set_title(f'XPL% - XPT% {mh} - {problema} - {clasificador} - {bss}')
#                     # axPER.set_ylabel("Percentage")
#                     # axPER.set_xlabel("Iteration")
#                     # axPER.legend(loc = 'upper right')
#                     # plt.savefig(f'{dirResultado}/Graficos/Percentage_{mh}_{problema}_{clasificador}_{bss}.pdf')
#                     # plt.close('all')
#                     # print(f'Grafico de exploracion y explotacion realizado para - {problema} - {clasificador} - {bss}')
                    
#                     print("-------------------------------------------------------------------------------------------------------")
                    
#                 os.remove('./Resultados/Transitorio/'+nombreArchivo+"_"+clasificador+'.csv')
            
archivoResumenFitness.close()
archivoResumenTimes.close()
archivoResumenFscore.close()
archivoResumenTotalFeaturesSelected.close()