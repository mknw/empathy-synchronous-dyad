import random
import dyads

#weights, symmetric except for adaptive connction
#Speed factorss symmetric[1:]
#Sc factor(R)
#steep(R)
#thresh(0-1)
#adcon mu tau alpha the rest is boolean not symmetric



def setrandomparams(length):
    params =[]
    for i in range(length):
        params.append(random.random())
    return params

def calcfitness(params):
        outputs = dyads.run_nets(params)
        x7= outputs['X7']
        del outputs['X7']
        A = sum(outputs.values())
        fitness = x7-A
        return fitness


weightparams = setrandomparams(11)
speedparams = setrandomparams(8)
combparams = setrandomparams(6)
for i in range(4):
    combparams[i] = combparams[i]*random.randint(1,100)     
all_params= [weightparams, speedparams,combparams]
print(calcfitness(all_params))
    
