import dyads
from pyswarm import pso
import pandas as pd
import numpy as np
import random

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
    outputs,x3sx7s_SSR,x3sx3ns_SSR,x3nsx7s_SSR,x3sx7ns_SSR,x5x9_SSR,end_val = dyads.compare_nets(net1,net2)
    x3= outputs['X3']
    x5= outputs['X5']
    x7= outputs['X7']+0.00000000000000000000000000000000000001
    x9= outputs['X9']+0.00000000000000000000000000000000000001
    del outputs['X7']
    A = sum(outputs.values())
   B = abs((0.8/X7s)-1)
   B2 = 10*(X7s**2)-16*X7s+6.4000001
   B3 = abs(2*X7s-1.6)
   C = abs((0.45/X7ns)-1)
   C2 = 10*(X7ns**2)-9*X7ns+2.0250001
   C3 = abs(2*X7ns-0.9)
    D = 1/x7
    E = 1/x9
   F = abs(x3-x4)+0.00000000000000000000000000000000000001
   G = abs(x7-x8)+0.00000000000000000000000000000000000001
   H = 1/F
   I = 1/x9
   J = x5
    K = (x3sx7s_SSR + x3sx3ns_SSR + x3nsx7s_SSR)/3
    L = 10/x3sx7ns_SSR
    M = x5
    N = x5x9_SSR
    O = 100/(((end_val['X7'][0]-end_val['X7'][1])**2)+0.00000000000000000000001)
    #hebbian X3 X4, X7 X8 *should as close to 0 also as high as possible between 0.3 and 0.8
    #excecution state similar
    #X3s,X7s,X3ns same and minimize X3ns
    #make function to calculate difference for each time step. (output in sse like values)
    #flip between one and zero for between %10
    #make function that check on each time step if the difference
    #the weights between X3s X4s and X7s and X8s should be the same as X3ns X4ns and minimize weight of x7ns x8ns
    #and the same with the states of X5 and X9
#    fitness = K+L+E+M+N
    fitness = A+D
    print(params)
#    print(E,K,L,M,N)
    print(fitness)
    return fitness

def pso_run():
    global init_val0
    global init_val1
    world = 1
    init_val0 = [
        np.array(['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']),
        pd.read_pickle('weights_df.pickle0'),
        pd.read_pickle('speed_factors_df.pickle0'),
        pd.read_pickle('comb_par_df.pickle0'),
        pd.read_pickle('adcon_par_df.pickle0'),
        [float(n) for n in[world, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            ]
    init_val1 = [
        np.array(['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']),
        pd.read_pickle('weights_df.pickle1'),
        pd.read_pickle('speed_factors_df.pickle1'),
        pd.read_pickle('comb_par_df.pickle1'),
        pd.read_pickle('adcon_par_df.pickle1'),
        [float(n) for n in[world, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            ]

    lb = [0.001]*20
    ub = [1]*20

    xopt, fopt = pso(calcfitness, lb, ub,swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, minstep=1e-6,
        minfunc=1e-06, debug=False)

    return(xopt,fopt)
