import random
import dyads

#weights, symmetric except for adaptive connction
#Speed factorss symmetric[1:]
#Sc factor(R)
#steep(R)
#thresh(0-1)
#adcon mu tau alpha the rest is boolean not symmetric



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

def calcfitness(net1,net2,params):
#        outputs = dyads.run_nets(params)
        net1 = dyads.update_net(net1, params)
        net2 = dyads.update_net(net2, params)
        outputs = dyads.compare_nets(net1,net2)
        x7= outputs['X7']
        del outputs['X7']
        A = sum(outputs.values())
        fitness = x7-A
        return fitness


def initiateanneal():
    weightparams = setrandomparams(7,0)
    speedparams = setrandomparams(4,0)
    combparams = setrandomparams(3,0)
    adconparams = setrandomparams(6,0)     
    all_params= [weightparams, speedparams,combparams,adconparams]
    return all_params

paramsa = initiateanneal()
soc,nonsoc = dyads.init_nets()

#found = False
iterate = 100
#old_fitness = calcfitness(params)
old_fitness = -5000
step_size = 0.05
heat = 25

while iterate >= 0:
    index = 0
    for paramset in paramsa:
        for param in range(len(paramset)):
            sign = random.choice(['-','+'])
            testparams = paramsa
            exec('new_value = testparams[index][param] %s step_size' %sign)
            if new_value <= 1 and new_value >= 0:
                testparams[index][param] = new_value
            else:
                continue
            fitness = calcfitness(soc,nonsoc,testparams)
#            fitness = 1
            accept = random.random()
            if accept > heat/100:
                if fitness > old_fitness:
#                    params = testparams
                    print ('better')
            else:
                print ('random')
#                params = testparams
#            dyads.update_net(soc, paramsa)
#            dyads.update_net(nonsoc, paramsa)
            print (paramsa)
            print (fitness)
        index+=1
    iterate -= 1
    heat -= 0.25
    
