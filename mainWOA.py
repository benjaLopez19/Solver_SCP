from Solver.solverFS import solverFS
from Solver.solverFSML import solverFSML
from Solver.solverSCP import solverSCP
from Solver.solverB import solverB
from Solver.solverMKP import solverMKP
# from Solver.solverEMPATIA import solverEMPATIA
from Solver.solverEMPATIAML import solverEMPATIAML
from Solver.solverEMPATIA_multiple import solverEMPATIA
from BD.sqlite import BD
import json
# problems = ['ionosphere.data']
bd = BD()

# data = bd.obtenerExperimento()
datas = bd.obtenerExperimentos()

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

ids = []

for data in datas:
    ds = []
    # print(data)
    id = int(data[0])
    id_instancia = int(data[8])
    datosInstancia = bd.obtenerInstancia(id_instancia)
    # print(datosInstancia)
    mh = data[1]
    parametrosMH = data[2]
    
    # print(id)
    # print(mh)

    instancia = datosInstancia[0][2]
    ds.append(parametrosMH.split(",")[2].split(":")[1].split("-")[0])
    ds.append(parametrosMH.split(",")[2].split(":")[1].split("-")[1])
    pop = int(parametrosMH.split(",")[1].split(":")[1])
    maxIter = int(parametrosMH.split(",")[0].split(":")[1])
    
    if mh == 'WOA':
        ids.append(id)
        bd.actualizarExperimento(id, 'ejecutando')

print(ids)
print('WOA')
print(1000)
print(pop)
print(ds)
print(instancia)

solverEMPATIA(ids, 'WOA', 1000, pop, ds, instancia)
    

    
print("-------------------------------------------------------")
print("-------------------------------------------------------")
print("Se han ejecutado todos los experimentos pendientes de WOA.")
print("-------------------------------------------------------")
print("-------------------------------------------------------")

