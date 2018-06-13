import random
import dyads
import numpy as np
import pandas as pd
import itertools

#weights, symmetric except for adaptive connction
#Speed factorss symmetric[1:]
#Sc factor(R)
#steep(R)
#thresh(0-1)
#adcon mu tau alpha the rest is boolean not symmetric

global init_val0
global init_val1

def setrandomparams(length, mirror):
    params =[]
    if mirror == 1:
        for i in range(int(length/2)):
            num = random.random()
            params.append(num)
        for i in range(int(length/2)):
            params.append(params[i])
    else:
        for i in range(length):
            params.append(random.random())
    return params

#def fitnessfunction(outputs):
#    

def calcfitness(params):
    global init_val0
    global init_val1
    wp = params[:7]
    sp = params[7:11]
    cp = params[11:14]
    ap = params[14:]
    formatted_params =[wp,sp,cp,ap]
    net1 = dyads.test_net(1,init_val1, formatted_params)
    net2 = dyads.test_net(0,init_val0, formatted_params)
    outputs = dyads.compare_nets(net1,net2)
    x7= outputs['X7']+0.00000000000000000000000000000000000001
    del outputs['X7']
    A = sum(outputs.values())
    fitness = A+(1/x7)
    print(params)
    print(fitness)
    return fitness


def initiateanneal():
    weightparams = setrandomparams(7,0)
    speedparams = setrandomparams(4,0)
    combparams = setrandomparams(3,0)
    adconparams = setrandomparams(6,0)     
    all_params=setrandomparams(20,0)
    return all_params

paramsa = initiateanneal()
#soc,nonsoc = dyads.init_nets(paramsa)


iterate = 100000
old_fitness = -5000
step_size = 0.05
heat = 25
solution = ''

init_val0 = [
    np.array(['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']),
    pd.read_pickle('weights_df.pickle0'),
    pd.read_pickle('speed_factors_df.pickle0'),
    pd.read_pickle('comb_par_df.pickle0'),
    pd.read_pickle('adcon_par_df.pickle0'),
    [float(n) for n in[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        ]
init_val1 = [
    np.array(['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']),
    pd.read_pickle('weights_df.pickle1'),
    pd.read_pickle('speed_factors_df.pickle1'),
    pd.read_pickle('comb_par_df.pickle1'),
    pd.read_pickle('adcon_par_df.pickle1'),
    [float(n) for n in[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        ]

#while iterate >= 0:
#    index = 0
#    for paramset in paramsa:
#        for param in range(len(paramset)):
#            print('new')
#            sign = random.choice(['-','+'])
#            testparams = paramsa
#            exec('new_value = testparams[index][param] %s step_size' %sign)
#            if new_value <= 1 and new_value >= 0:
#                testparams[index][param] = new_value
#            else:
#                continue
#            fitness = calcfitness(testparams)
#            accept = random.random()
#            if fitness < old_fitness:
#                print ('better')
#                paramsa = testparams
#                old_fitness = fitness
#            else:
#                if accept < heat/100:
#                    print ('random')
#                    paramsa = testparams
#                    old_fitness = fitness
#        index+=1
#    iterate -= 1
#    heat -= 0.25
    
while iterate >=0 or solution != 'found':
    br = 0
    neighbour = [0]*20
    amount = random.randint(1,20)
    for i in range(amount):
        polarity = random.choice([-1,1])
        neighbour[random.randint(0,19)] = step_size*polarity
    summing = [paramsa, neighbour]
    direction = [sum(i) for i in zip(*summing)]
    for i in direction:
        if i <= 0 or i > 1:
            br = 1
            break
    if br == 1:
        continue
    fitness = calcfitness(direction)
    accept = random.random()
    if fitness < old_fitness:
        print ('better')
        paramsa = direction
        old_fitness = fitness
    else:
        if accept < heat/100:
            print ('random')
            paramsa = direction
            old_fitness = fitness
        continue
    if fitness < 1:
        solution = 'found'
    iterate -= 1
    heat -= heat*0.01
    print ('heat'+str(heat))
    


