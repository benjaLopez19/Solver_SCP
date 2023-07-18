import numpy as np


class mkp:
    def __init__(self, instance, tipo):
        self.__cost           = None
        self.__weight         = None
        self.__restriccions   = None
        self.__optimo         = None
        self.__elements       = None
        self.__knapsack       = None
        if tipo == 1:
            self.lecturaArchivoMknap1(instance)
        if tipo == 2:
            self.lecturaArchivoMknap2(instance)
            
    def getCost(self):
        return self.__cost
    def setCost(self, cost):
        self.__cost = cost
    def getWeight(self):
        return self.__weight
    def setWeight(self, weight):
        self.__weight = weight
    def getRestriccions(self):
        return self.__restriccions
    def setRestriccions(self, restriccions):
        self.__restriccions = restriccions
    def getOptimo(self):
        return self.__optimo
    def setOptimo(self, optimo):
        self.__optimo = optimo
    def getElements(self):
        return self.__elements
    def setElements(self, elements):
        self.__elements = elements
    def getKnapsack(self):
        return self.__knapsack
    def setKnapsack(self, knapsack):
        self.__knapsack = knapsack
    
    def lecturaArchivoMknap1(self, instance):
        with open('./Problem/MKP/Instances/'+instance+".txt",'r',encoding = 'utf-8') as f:
            line = f.readline().split(" ")
            
            elements    = int(line[0])
            knapsack    = int(line[1])
            optimo      = float(line[2])
            
            cost = np.zeros(shape=elements)  
            line = f.readline()
            countDim = 1

            while line != "" and countDim <= elements:
                values = line.split()
                for i in range(len(values)):
                    cost[countDim-1] = float(values[i])
                    countDim +=1
                line = f.readline()
            
            weight = []
            for i in range(knapsack):
                countDim = 1
                aux = np.zeros(shape=elements)
                while line != "" and countDim <= elements:
                    values = line.split()
                    for i in range(len(values)):
                        aux[countDim-1] = float(values[i])
                        countDim +=1
                    line = f.readline()
                weight.append(aux)
            weight = np.array(weight)

            restriccions = np.zeros(knapsack)
            countKnap = 1
            
            while line != "" and countKnap <= knapsack:
                values = line.split()
                for i in range(len(values)):
                    restriccions[countKnap-1] = float(values[i])
                    countKnap +=1
                line = f.readline()
                
            self.setCost(cost)
            self.setWeight(weight)
            self.setRestriccions(restriccions)
            self.setOptimo(optimo)
            self.setElements(elements)
            self.setKnapsack(knapsack)
            
    def lecturaArchivoMknap2(self, instance):
        with open('./Problem/MKP/Instances/'+instance+".txt",'r',encoding = 'utf-8') as f:
            line = f.readline().split(" ")
            
            knapsack    = int(line[0])
            elements    = int(line[1])
            
            cost = np.zeros(shape=elements)  
            line = f.readline()
            countDim = 1

            while line != "" and countDim <= elements:
                values = line.split()
                for i in range(len(values)):
                    cost[countDim-1] = float(values[i])
                    countDim +=1
                line = f.readline()
            
            restriccions = np.zeros(knapsack)
            countKnap = 1
            
            while line != "" and countKnap <= knapsack:
                values = line.split()
                for i in range(len(values)):
                    restriccions[countKnap-1] = float(values[i])
                    countKnap +=1
                line = f.readline()
            
            weight = []
            for i in range(knapsack):
                countDim = 1
                aux = np.zeros(shape=elements)
                while line != "" and countDim <= elements:
                    values = line.split()
                    for i in range(len(values)):
                        aux[countDim-1] = float(values[i])
                        countDim +=1
                    line = f.readline()
                weight.append(aux)
            weight = np.array(weight)
            
            line = f.readline()
            optimo = float(line)
             
            self.setCost(cost)
            self.setWeight(weight)
            self.setRestriccions(restriccions)
            self.setOptimo(optimo)
            self.setElements(elements)
            self.setKnapsack(knapsack)
            
    def repairSolution(self, solution):
        r = np.zeros(shape=self.getKnapsack())
        i = 0
        for w in self.getWeight():
            r[i] = np.sum(solution * w)
            i += 1
        
        j = self.getElements() - 1
        while j > -1:
            for i in range(self.getKnapsack()):
                if solution[j] == 1 and r[i] > self.getRestriccions()[i]:
                    solution[j] = 0
                    for k in range(self.getKnapsack()):
                        r[k] = r[k] - self.getWeight()[k][j]
            j -=1
        
        for j in range(self.getElements()):
            agregar = True
            for i in range(self.getKnapsack()):
                suma = r[i] + self.getWeight()[i][j]
                if solution[j] == 0 and suma > self.getRestriccions()[i]:
                    agregar = False
            
            if agregar:
                solution[j] = 1
                for k in range(self.getKnapsack()):
                    r[k] = r[k] + self.getWeight()[k][j]
        
        return solution
    
    def calcule_fitness(self, solution):
        return np.sum(solution * self.getCost())
    
    def test_factibilidad(self, solution):
        feasible = True
        if np.sum(solution) == 0:
            feasible = False
        else:
            i = 0
            restrictions = self.getRestriccions()
            for w in self.getWeight():
                sum = np.sum(solution * w)
                if sum > restrictions[i]:
                    feasible = False
                i += 1
            
        return feasible

