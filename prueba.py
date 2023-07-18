import random
import numpy as np


# crossover operator
# def crossover(parent1, parent2, prob_crossover):
       
#     pivot = int(np.round( (len(parent1) * prob_crossover) , 0))
    
#     print(pivot)
    
#     child1 = parent1[:pivot] + parent2[pivot:]
#     child2 = parent2[:pivot] + parent1[pivot:]
    
#     return child1, child2

# hijo1 = np.array([1,1,1,1,1,1,1,1])
# hijo2 = np.array([0,0,0,0,0,0,0,0])

# print(hijo1)
# print(hijo2)

# prob_crossover = 0.6

# hijo1, hijo2 = crossover(hijo1.tolist(),hijo2.tolist(),prob_crossover)

# print(hijo1)
# print(hijo2)

param = "iter:500,pop:20,DS:V4-STD,cros:0.6;mut:0.01"
paramGA = param.split(",")[3]
print(float(paramGA.split(";")[0].split(":")[1]))
print(float(paramGA.split(";")[1].split(":")[1]))