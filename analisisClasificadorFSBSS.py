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
                                  "EMPATIA-9"
                                  ''')
# clasificadores = ["KNN","RandomForest","Xgboost"]
clasificadores = [""]

binarizaciones = ["10","20","30","40","50","60","70","80","90","100"]

# binarizaciones = ["S4-STD","V4-STD","X4-STD","Z4-STD","S4-COM","V4-COM","X4-COM","Z4-COM","S4-ELIT","V4-ELIT","X4-ELIT","Z4-ELIT"]

dirResultado = './Resultados/'

# --------------------------------------------------------------------------------------- #

print(instancias)
for clasificador in clasificadores:
    
    archivoResumen = open(f'{dirResultado}resumen_{clasificador}.csv', 'w')
    archivoResumenFitness = open(f'{dirResultado}resumen_fitness_{clasificador}.csv', 'w')
    archivoResumenFscore = open(f'{dirResultado}resumen_fscore_{clasificador}.csv', 'w')
    archivoResumenTimes = open(f'{dirResultado}resumen_times_{clasificador}.csv', 'w')
    archivoResumenTotalFeaturesSelected = open(f'{dirResultado}resumen_total_feature_selected_{clasificador}.csv', 'w')
    
    # for bss in binarizaciones:
    for instancia in instancias:
        # archivoResumenFitness.write(f''' ,{bss}, ,''')
        # # archivoResumenFscore.write(f''' ,{bss}, ,''')
        # archivoResumenTimes.write(f''' ,{bss}, ,''')
        
        archivoResumenFitness.write(f''' ,{instancia[1]}, ,''')
        archivoResumenFscore.write(f''' ,{instancia[1]}, ,''')
        archivoResumenTimes.write(f''' ,{instancia[1]}, ,''')
        archivoResumenTotalFeaturesSelected.write(f''' ,{instancia[1]},''')
    
    # archivoResumenFitness.write(' \n')
    # archivoResumenFitness.write('instance')
    
    archivoResumen.write('BSS')
    
    archivoResumenFitness.write(' \n')
    archivoResumenFitness.write('BSS')
    
    archivoResumenFscore.write(' \n')
    archivoResumenFscore.write('BSS')
    
    # archivoResumenTimes.write(' \n')
    # archivoResumenTimes.write('instance')
    
    archivoResumenTimes.write(' \n')
    archivoResumenTimes.write('BSS')
    
    archivoResumenTotalFeaturesSelected.write(' \n')
    archivoResumenTotalFeaturesSelected.write('BSS')
    
    # for bss in binarizaciones:
    for instancia in instancias:
        archivoResumen.write(f',best,avg,dev-std,min,avg,dev-std,avg,dev-std')
        archivoResumenFitness.write(f',best,avg,dev-std')
        archivoResumenFscore.write(f',best,avg,dev-std')
        archivoResumenTimes.write(f',min,avg,dev-std')
        archivoResumenTotalFeaturesSelected.write(f',avg,dev-std')
    
    archivoResumen.write('\n')
    archivoResumenFitness.write('\n')
    archivoResumenFscore.write('\n')
    archivoResumenTimes.write('\n')
    archivoResumenTotalFeaturesSelected.write('\n')
    
    primero = True
    
    # for instancia in instancias:
    for bss in binarizaciones:
        
        if primero:
            # archivoResumenFitness.write(instancia[1])
            # # archivoResumenFscore.write(instancia[1])
            # archivoResumenTimes.write(instancia[1])
            
            archivoResumen.write(bss)
            archivoResumenFitness.write(bss)
            archivoResumenFscore.write(bss)
            archivoResumenTimes.write(bss)
            archivoResumenTotalFeaturesSelected.write(bss)
            primero = False
        else:
            # archivoResumenFitness.write('\n')
            # archivoResumenFitness.write(instancia[1])
            # # archivoResumenFscore.write('\n')
            # # archivoResumenFscore.write(instancia[1])
            # archivoResumenTimes.write('\n')
            # archivoResumenTimes.write(instancia[1])
            
            archivoResumen.write('\n')
            archivoResumen.write(bss)
            archivoResumenFitness.write('\n')
            archivoResumenFitness.write(bss)
            archivoResumenFscore.write('\n')
            archivoResumenFscore.write(bss)
            archivoResumenTimes.write('\n')
            archivoResumenTimes.write(bss)
            archivoResumenTotalFeaturesSelected.write('\n')
            archivoResumenTotalFeaturesSelected.write(bss)
            
        for instancia in instancias:
            
            print("-------------------------------------------------------------------------------------------------------")
            print(f'''{clasificador} - {instancia[1]} - {bss}''')
            
            blob = bd.obtenerArchivosBSSClasificador(instancia[1],"",bss, clasificador)
            
            fitnessPSA = []
            fscorePSA = []
            timesPSA = []
            totalFeatureSelectedPSA = []
            
            for d in blob:
                
                nombreArchivo = d[6]
                archivo = d[7]
                mh = d[1]
                print(mh)
                problema = d[5]
                
                direccionDestiono = './Resultados/Transitorio/'+nombreArchivo+"_"+clasificador+'.csv'
                
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
                
                
                if mh == 'GA':
                    fitnessPSA.append(fitness[ultimo])
                    fscorePSA.append(f1Score[ultimo])
                    timesPSA.append(np.round(np.sum(time),1))
                    totalFeatureSelectedPSA.append(tfs[ultimo])
                    
                os.remove('./Resultados/Transitorio/'+nombreArchivo+"_"+clasificador+'.csv')
                
            archivoResumen.write(F''',{np.max(fscorePSA)},{np.round(np.average(fscorePSA),3)},{np.round(np.std(fscorePSA),3)},{np.min(timesPSA)},{np.round(np.average(timesPSA),1)},{np.round(np.std(timesPSA),1)},{np.round(np.average(totalFeatureSelectedPSA),1)},{np.round(np.std(totalFeatureSelectedPSA),1)}''')
            archivoResumenFitness.write(F''',{np.min(fitnessPSA)},{np.round(np.average(fitnessPSA),3)},{np.round(np.std(fitnessPSA),3)}''')
            archivoResumenFscore.write(F''',{np.max(fscorePSA)},{np.round(np.average(fscorePSA),3)},{np.round(np.std(fscorePSA),3)}''')
            archivoResumenTimes.write(F''',{np.min(timesPSA)},{np.round(np.average(timesPSA),1)},{np.round(np.std(timesPSA),1)}''')
            archivoResumenTotalFeaturesSelected.write(F''',{np.round(np.average(totalFeatureSelectedPSA),1)},{np.round(np.std(totalFeatureSelectedPSA),1)}''')
            
            
            blob = bd.obtenerMejoresArchivosconClasificadorBSS(instancia[1],"",clasificador,bss)
            
            bestFitnessPSA      = fitness
            bestTimePSA         = time
            bestFscorePSA       = f1Score
            bestDIVPSA          = div 
            bestXPLPSA          = xpl
            bestXPTPSA          = xpt
            bestfsPSA           = tfs
            
            
            for d in blob:
                
                nombreArchivo = d[6]
                archivo = d[7]
                mh = d[1]
                problema = d[5]
                
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
                
                if mh == 'GA':
                    bestFitnessPSA      = fitness
                    bestTimePSA         = time
                    bestFscorePSA       = f1Score
                    bestDIVPSA          = div 
                    bestXPLPSA          = xpl
                    bestXPTPSA          = xpt
                    bestfsPSA           = tfs
                
                    # print("------------------------------------------------------------------------------------------------------------")
                    figPER, axPER = plt.subplots()
                    axPER.plot(iteraciones, bestFitnessPSA, color="g", label="GA_"+bss+": "+str(np.min(bestFitnessPSA)))
                    # axPER.set_title(f'Coverage - {problema} - {clasificador} - {bss}')
                    axPER.set_title(f'Coverage - {problema} - {bss}')
                    axPER.set_ylabel("Fitness")
                    axPER.set_xlabel("Iteration")
                    axPER.legend(loc = 'upper right')
                    plt.savefig(f'{dirResultado}/Best/fitness_{problema}_{clasificador}_{bss}.pdf')
                    plt.close('all')
                    print(f'Grafico de fitness realizado - {problema} - {clasificador} - {bss}')
                    
                    # figPER, axPER = plt.subplots()
                    # axPER.plot(iteraciones, bestTimePSA, color="g", label="PSA")
                    # axPER.set_title(f'Time (s) - {problema} - {clasificador} - {bss}')
                    # axPER.set_ylabel("Time (s)")
                    # axPER.set_xlabel("Iteration")
                    # axPER.legend(loc = 'upper right')
                    # plt.savefig(f'{dirResultado}/Best/time_{problema}_{clasificador}_{bss}.pdf')
                    # plt.close('all')
                    # print(f'Grafico de time realizado - {problema} - {clasificador} - {bss}')
                    
                    figPER, axPER = plt.subplots()
                    axPER.plot(iteraciones, bestFscorePSA, color="g", label="GA_"+bss+": "+str(np.max(bestFscorePSA)))
                    # axPER.set_title(f'f-score - {problema} - {clasificador} - {bss}')
                    axPER.set_title(f'f-score - {problema} - {bss}')
                    axPER.set_ylabel("f-score")
                    axPER.set_xlabel("Iteration")
                    axPER.legend(loc = 'lower right')
                    plt.savefig(f'{dirResultado}/Best/fscore_{problema}_{clasificador}_{bss}.pdf')
                    plt.close('all')
                    print(f'Grafico de f-score realizado - {problema} - {clasificador} - {bss}')
                    
                    # fig , ax = plt.subplots()
                    # ax.plot(iteraciones,bestDIVPSA)
                    # ax.set_title(f'Diversity {mh} - {problema} - {clasificador} - {bss}')
                    # ax.set_ylabel("Diversity")
                    # ax.set_xlabel("Iteration")
                    # plt.savefig(f'{dirResultado}/Graficos/Diversity_{mh}_{problema}_{clasificador}_{bss}.pdf')
                    # plt.close('all')
                    # print(f'Grafico de diversidad realizado - {problema} - {clasificador} - {bss}')
                    
                    fig , ax = plt.subplots()
                    # ax.plot(iteraciones,bestfsPSA, color= "g", label="GA_"+bss)
                    ax.plot(iteraciones, bestfsPSA, color="g", label="GA_"+bss+": "+str(bestfsPSA[len(iteraciones) - 1]))
                    # ax.set_title(f'Total Feature Selected {mh} - {problema} - {clasificador} - {bss}')
                    ax.set_title(f'Total Feature Selected {mh} - {problema} - {bss}')
                    ax.set_ylabel("Total Feature Selected")
                    ax.set_xlabel("Iteration")
                    ax.legend(loc = 'upper right')
                    plt.savefig(f'{dirResultado}/Graficos/TFS_{mh}_{problema}_{clasificador}_{bss}.pdf')
                    plt.close('all')
                    print(f'Grafico de diversidad realizado - {problema} - {clasificador} - {bss}')
                    
                    # figPER, axPER = plt.subplots()
                    # axPER.plot(iteraciones, bestXPLPSA, color="r", label=r"$\overline{XPL}$"+": "+str(np.round(np.mean(bestXPLPSA), decimals=2))+"%")
                    # axPER.plot(iteraciones, bestXPTPSA, color="b", label=r"$\overline{XPT}$"+": "+str(np.round(np.mean(bestXPTPSA), decimals=2))+"%")
                    # axPER.set_title(f'XPL% - XPT% {mh} - {problema} - {clasificador} - {bss}')
                    # axPER.set_ylabel("Percentage")
                    # axPER.set_xlabel("Iteration")
                    # axPER.legend(loc = 'upper right')
                    # plt.savefig(f'{dirResultado}/Graficos/Percentage_{mh}_{problema}_{clasificador}_{bss}.pdf')
                    # plt.close('all')
                    # print(f'Grafico de exploracion y explotacion realizado para - {problema} - {clasificador} - {bss}')
                    
                    print("-------------------------------------------------------------------------------------------------------")
                    
                os.remove('./Resultados/Transitorio/'+nombreArchivo+"_"+clasificador+'.csv')

    archivoResumen.close()            
    archivoResumenFitness.close()
    archivoResumenTimes.close()
    archivoResumenFscore.close()
    archivoResumenTotalFeaturesSelected.close()