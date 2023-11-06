from Solver.solverSCP import solverSCP
from Solver.solverB import solverB
from BD.sqlite import BD
import json
import time
import multiprocessing

def procesarExperimento(data):
    bd = BD()
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
    
    if problema == 'SCP':
        #bd.actualizarExperimento(id, 'ejecutando')
        repair = parametrosMH.split(",")[3].split(":")[1]
        ds.append(parametrosMH.split(",")[2].split(":")[1].split("-")[0])
        ds.append(parametrosMH.split(",")[2].split(":")[1].split("-")[1])
        
        parMH = parametrosMH.split(",")[4]
        print(parMH)
        solverSCP(id, mh, maxIter, pop, instancia, ds, repair, parMH,False)
     
    if problema == 'BEN':
        #bd.actualizarExperimento(id, 'ejecutando')
        lb =  float(parametrosInstancia.split(",")[0].split(":")[1])
        ub =  float(parametrosInstancia.split(",")[1].split(":")[1])
        dim =  int(parametrosInstancia.split(",")[2].split(":")[1])
        solverB(id, mh, maxIter, pop, instancia, lb, ub, dim)
        
    data = bd.obtenerExperimento()

# problems = ['ionosphere.data']
if __name__ == "__main__":
    bd = BD()
    #num de hilos a usar
    n = 5
    experimentos = []
    for i in range(n):
        data = bd.obtenerExperimento()
        if(len(data)==0):
            break
        experimentos.append(data)
        id = int(data[0][0])
        bd.actualizarExperimento(id, 'ejecutando')
    #print(experimentos)

    #exit(1)
    inicio = time.time()
    batch = 0
    while len(experimentos) > 0:
        init_batch = time.time() 
         # Create a Pool of processes (the number of processes is determined automatically)
        with multiprocessing.Pool() as pool:
            # Map the function to the list of numbers to calculate squares in parallel
            pool.map(procesarExperimento, experimentos)

        # Wait for all child processes to finish
        pool.close()
        pool.join()

        print("-------------------------------------------------------")
        print("Terminó batch:",batch)
        print("Tiempo total:",str(time.time()-init_batch))
        print("-------------------------------------------------------")

        experimentos = []
        for i in range(n):
            data = bd.obtenerExperimento()
            if(len(data)==0):
                break
            experimentos.append(data)
            id = int(data[0][0])
            bd.actualizarExperimento(id, 'ejecutando')
        batch +=1
        
        
        
    print("-------------------------------------------------------")
    print("-------------------------------------------------------")
    print("Se han ejecutado todos los experimentos pendientes.")
    print("Tiempo total:",str(time.time()-inicio))
    print("-------------------------------------------------------")
    print("-------------------------------------------------------")

