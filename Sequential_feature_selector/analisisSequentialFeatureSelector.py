import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

contenido = os.listdir("./")

sfs_tfs = []
sfs_times = []
sfs_f_scores = []

for archivo in contenido:
    
    if archivo.endswith('.csv'):
        if "performance_merge_EMPATIA-9" in archivo:
            
            doc = pd.read_csv(f'''./{archivo}''')
            
            tfs = doc["tfs"]
            time = doc["time"]
            f_score = doc["f-score"]
            
            pos = np.argmax(f_score)
            
            sfs_tfs.append(tfs[pos])
            sfs_times.append(np.sum(time))
            sfs_f_scores.append(f_score[pos])
            
            figPER, axPER = plt.subplots()
            axPER.plot(tfs, f_score, color="g", label="SFS: "+str(np.max(f_score)))
            # axPER.set_title(f'f-score - {problema} - {clasificador} - {bss}')
            axPER.set_title(f'f-score - EMPATIA-9 - SFS')
            axPER.set_ylabel("f-score")
            axPER.set_xlabel("total features selected")
            axPER.legend(loc = 'lower right')
            plt.savefig(f'./fscore.pdf')
            plt.close('all')
            print(f'Grafico de f-score realizado')
            
            
print(f'''{np.max(sfs_f_scores)},{np.average(sfs_f_scores)},{np.std(sfs_f_scores)},{np.average(sfs_tfs)},{np.std(sfs_tfs)},{np.round(np.min(sfs_times),1)},{np.round(np.average(sfs_times),1)},{np.round(np.std(sfs_times),1)}''')

