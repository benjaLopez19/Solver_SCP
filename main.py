from Solver.solverFS import solverFS
from Solver.solverFSML import solverFSML
from Solver.solverSCP import solverSCP
from Solver.solverB import solverB
from Solver.solverMKP import solverMKP
from Solver.solverEMPATIA import solverEMPATIA
from Solver.solverEMPATIAML import solverEMPATIAML
from Solver.solver_EMPATIA_voluntaria import solverEMPATIA_Voluntaria
from BD.sqlite import BD
import json
# problems = ['ionosphere.data']
bd = BD()

data = bd.obtenerExperimento()

id              = 0
instancia       = ''
problema        = ''
mh              = ''
parametrosMH    = ''
maxIter         = 0
pop             = 0
ds              = []
clasificador    = ''
parametrosC     = '' 

pruebas = 1
while len(data) > 0: 
# while pruebas == 1:
    print("-------------------------------------------------------------------------------------------------------")
    print(data)
    
    id = int(data[0][0])
    id_instancia = int(data[0][8])
    datosInstancia = bd.obtenerInstancia(id_instancia)
    print(datosInstancia)
    
    problema = datosInstancia[0][1]
    instancia = datosInstancia[0][2]
    parametrosInstancia = datosInstancia[0][4]
    mh = data[0][1]
    parametrosMH = data[0][2]
    ml = data[0][3]

    
    maxIter = int(parametrosMH.split(",")[0].split(":")[1])
    pop = int(parametrosMH.split(",")[1].split(":")[1])
    ds = []
    
    if problema == 'FS':
        bd.actualizarExperimento(id, 'ejecutando')
        
        if len(ml) > 1:
            parametrosML = json.loads(data[0][4])
            if instancia == 'EMPATIA' or instancia == 'EMPATIA-2' or instancia == 'EMPATIA-3' or instancia == 'EMPATIA-4' or instancia == 'EMPATIA-5' or instancia == 'EMPATIA-6' or instancia == 'EMPATIA-7' or instancia == 'EMPATIA-8' or instancia == 'EMPATIA-9' or instancia == 'EMPATIA-10' or instancia == 'EMPATIA-11' or instancia == 'EMPATIA-12': 
                solverEMPATIAML(id, mh, maxIter, pop, instancia, parametrosML, ml)
            else:
                clasificador = data[0][5]
                parametrosC = data[0][6]
                solverFSML(id, mh, maxIter, pop, instancia, clasificador, parametrosC, parametrosML, ml)
        else:
            
            ds.append(parametrosMH.split(",")[2].split(":")[1].split("-")[0])
            ds.append(parametrosMH.split(",")[2].split(":")[1].split("-")[1])
            
            parMH = parametrosMH.split(",")[3]
            
            if instancia == 'EMPATIA' or instancia == 'EMPATIA-2' or instancia == 'EMPATIA-3' or instancia == 'EMPATIA-4' or instancia == 'EMPATIA-5' or instancia == 'EMPATIA-6' or instancia == 'EMPATIA-7' or instancia == 'EMPATIA-8' or instancia == 'EMPATIA-9' or instancia == 'EMPATIA-10' or instancia == 'EMPATIA-11' or instancia == 'EMPATIA-12': 
                solverEMPATIA(id, mh, maxIter, pop, ds, instancia, parMH)
                
            # if "EMPATIA-V" in instancia:
            #     solverEMPATIA_Voluntaria(id, mh, maxIter, pop, ds, instancia)
            # else:
            #     clasificador = data[0][5]
            #     parametrosC = data[0][6]
            #     solverFS(id, mh, maxIter, pop, instancia, ds, clasificador, parametrosC)
    
    if problema == 'SCP':
        bd.actualizarExperimento(id, 'ejecutando')
        repair = parametrosMH.split(",")[3].split(":")[1]
        ds.append(parametrosMH.split(",")[2].split(":")[1].split("-")[0])
        ds.append(parametrosMH.split(",")[2].split(":")[1].split("-")[1])
        solverSCP(id, mh, maxIter, pop, instancia, ds, repair)
    
    if problema == 'MKP':
        bd.actualizarExperimento(id, 'ejecutando')
        ds.append(parametrosMH.split(",")[2].split(":")[1].split("-")[0])
        ds.append(parametrosMH.split(",")[2].split(":")[1].split("-")[1])
        solverMKP(id, mh, maxIter, pop, instancia, ds)
    
    if problema == 'BEN':
        bd.actualizarExperimento(id, 'ejecutando')
        lb =  float(parametrosInstancia.split(",")[0].split(":")[1])
        ub =  float(parametrosInstancia.split(",")[1].split(":")[1])
        dim =  int(parametrosInstancia.split(",")[2].split(":")[1])
        solverB(id, mh, maxIter, pop, instancia, lb, ub, dim)
        
    data = bd.obtenerExperimento()
    
    print(data)
    
    
    pruebas += 1
    
print("-------------------------------------------------------")
print("-------------------------------------------------------")
print("Se han ejecutado todos los experimentos pendientes.")
print("-------------------------------------------------------")
print("-------------------------------------------------------")

