import dyads
from pyswarm import pso
import pandas as pd
import numpy as np
#weights, symmetric except for adaptive connction
#Speed factorss symmetric[1:]
#Sc factor(R)
#steep(R)
#thresh(0-1)
#adcon mu tau alpha the rest is boolean not symmetric

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
    outputs, X7s, X7ns = dyads.compare_nets(net1,net2)
    x7= outputs['X7']+0.00000000000000000000000000000000000001
    del outputs['X7']
    A = sum(outputs.values())
    B = abs((0.8/X7s)-1)
    B2 = 10*(X7s**2)-16*X7s+6.4000001
    B3 = abs(2*X7s-1.6)
    C = abs((0.45/X7ns)-1)
    C2 = 10*(X7ns**2)-9*X7ns+2.0250001
    C3 = abs(2*X7ns-0.9)
    D = 1/x7
    E = x7*-1
    fitness = A+B3+C3+D
    print(params)
    print(fitness)
    return fitness

global init_val0
global init_val1

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

lb = [0.001]*20
ub = [1]*20

xopt, fopt = pso(calcfitness, lb, ub,swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, minstep=1e-6,
    minfunc=1e-05, debug=False)

print (xopt,fopt)
