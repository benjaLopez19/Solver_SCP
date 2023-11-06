if __name__ == "__main__":
    bd = BD()
    #num de hilos a usar
    n = 6
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

