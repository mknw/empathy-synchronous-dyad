# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:06:42 2018

@author: Mike
"""

#import pandas as pd
#import numpy as np
#import networkx as nx
#from states_update import states_update
#from edges_update import edges_update
from import_data import syncNet
import numpy as np


#"""Create and prepare nx graphs through syncNet methods from import_data.py"""
#
### social condition
#soc_syncNet = syncNet(name="soc_syncNet", file="data/socialtapping_V4-final.xlsx")
#
#soc_syncNet.import_model()
##soc_syncNet.input_weights()
#soc_syncNet.build_dyad()
#soc_syncNet.plug_parameters()
#soc_syncNet.record_interaction(time=450)
##soc_syncNet.plot_activation()
##soc_syncNet.plot_weights()
#
#
### nonsocial condition
#nonsoc_syncNet = syncNet(name="nonsoc_syncNet", file="data/nonsocialtapping_V4-final.xlsx")
#
#nonsoc_syncNet.import_model()
##nonsoc_syncNet.input_weights()
#nonsoc_syncNet.build_dyad()
#nonsoc_syncNet.plug_parameters()
#nonsoc_syncNet.record_interaction(time=450)
##nonsoc_syncNet.plot_activation()
##nonsoc_syncNet.plot_weights()
#
#
#### assign NX GRAPHS created ( to  simplify following  code )
#soc_dyad = soc_syncNet.dyad
#
#nonsoc_dyad = nonsoc_syncNet.dyad
#
#
##"""edge data for keys of interest"""
##
##keysOI = ['hebbian', 'persistence', 'speed_factor', 'weight']
##for k in keysOI:
##   print(k, soc_dyad.get_edge_data('X3', 'X4')[0][k])
##
##
##"""node data"""
##
##soc_dyad.node['X3']

# =============================================================================
# #   - simulations model SSR comparison
# =============================================================================


#SSR_dict = {}

#for vrtx in soc_dyad.nodes():
#   vrtx_SSR = 0
#   social_actTL = list(soc_dyad.node[vrtx]['activityTimeLine'].values())
#   nonsocial_actTL = list(nonsoc_dyad.node[vrtx]['activityTimeLine'].values())
#   act_tuples = zip(social_actTL, nonsocial_actTL)
#   for act in range(len(social_actTL)):
#      vrtx_SSR += (social_actTL[act] - nonsocial_actTL[act])**2
#   SSR_dict[vrtx] = vrtx_SSR
#
#print("Sum of Squared Residuals between simulations for each state: ")
#print(SSR_dict)

def ssr(soc_dyad, nonsoc_dyad):
    x5ls = []
    x5x9_SSR = 0
    end_val ={}
    SSR_dict = {}
    weiSSR_dict ={}
    social_weiTL_x3x4 = list(soc_dyad.get_edge_data('X3', 'X4')[0]['weightTimeLine'].values())
    nonsocial_weiTL_x3x4 = list(nonsoc_dyad.get_edge_data('X3', 'X4')[0]['weightTimeLine'].values())
    social_weiTL_x7x8 = list(soc_dyad.get_edge_data('X7', 'X8')[0]['weightTimeLine'].values())
    nonsocial_weiTL_x7x8 = list(nonsoc_dyad.get_edge_data('X7', 'X8')[0]['weightTimeLine'].values())
    social_weiTL_x9x3 = list(soc_dyad.get_edge_data('X9', 'X3')[0]['weightTimeLine'].values())
    nonsocial_weiTL_x9x3 = list(nonsoc_dyad.get_edge_data('X9', 'X3')[0]['weightTimeLine'].values())
    social_weiTL_x5x7 = list(soc_dyad.get_edge_data('X5', 'X7')[0]['weightTimeLine'].values())
    nonsocial_weiTL_x5x7 = list(nonsoc_dyad.get_edge_data('X5', 'X7')[0]['weightTimeLine'].values())
    

    weiSSR_dict['x3x4sx7x8s'] = np.sum((np.array(social_weiTL_x3x4)[:] - np.array(social_weiTL_x7x8)[:]) ** 2)
    weiSSR_dict['x3x4sx3x4ns'] = np.sum( (np.array(social_weiTL_x3x4)[:] - np.array(nonsocial_weiTL_x3x4)[:]) ** 2)
    weiSSR_dict['x3x4nsx7x8s'] = np.sum( (np.array(nonsocial_weiTL_x3x4)[:] - np.array(social_weiTL_x7x8)[:]) ** 2)
    weiSSR_dict['x3x4sx7x8ns'] = np.sum( (np.array(social_weiTL_x3x4)[:] - np.array(nonsocial_weiTL_x7x8)[:]) ** 2)
    weiSSR_dict['x7x8sx7x8ns'] = np.sum( (np.array(social_weiTL_x7x8)[:] - np.array(nonsocial_weiTL_x7x8)[:]) ** 2)
    weiSSR_dict['x9x3sx9x3ns'] = np.sum( (np.array(social_weiTL_x9x3)[:] - np.array(nonsocial_weiTL_x9x3)[:]) ** 2)
    weiSSR_dict['x5x7sx5x7ns'] = np.sum( (np.array(social_weiTL_x5x7)[:] - np.array(nonsocial_weiTL_x5x7)[:]) ** 2)
    weiSSR_dict['x9x3sx5x7ns'] = np.sum( (np.array(social_weiTL_x9x3)[:] - np.array(nonsocial_weiTL_x5x7)[:]) ** 2)
    
    
    for vrtx in soc_dyad.nodes():
       social_actTL = np.array(list(soc_dyad.node[vrtx]['activityTimeLine'].values()))
       nonsocial_actTL = np.array(list(nonsoc_dyad.node[vrtx]['activityTimeLine'].values()))

       vrtx_SSR = np.sum( (social_actTL - nonsocial_actTL) ** 2)
       SSR_dict[vrtx] = vrtx_SSR
       if vrtx == 'X5':
           x5ls = social_actTL
       if vrtx == 'X9':
           for act in range(len(social_actTL)):
               x5x9_SSR += (social_actTL[act] - x5ls[act])**2
       end_val[vrtx] = (social_actTL[-1],nonsocial_actTL[-1])
#    print (SSR_dict)
    return(SSR_dict,weiSSR_dict,x5x9_SSR,end_val)
    
def ssr_within_dyad(dyad):
    p_ssr= 0
    dyad = dyad.dyad
    vrtxs = []
    for vrtx in dyad.nodes():
       vrtxs.append(vrtx)
    vrtxs = vrtxs[1:]
    for vrtx in range(len(vrtxs[1:5])):
       p1_actTL = np.array(list(dyad.node[vrtxs[vrtx]]['activityTimeLine'].values()))
       p2_actTL = np.array(list(dyad.node[vrtxs[vrtx+4]]['activityTimeLine'].values()))
       for act in range(len(p1_actTL)):
           p_ssr += (p1_actTL[act]-p2_actTL[act])**2
    return p_ssr

def input_params(net, params):
    net.input_weights(params[0])
    net.input_speed_factors(params[1])
    net.input_comb_par(params[2])
    net.input_adcon_par(params[3])

def init_nets(params):
    soc_syncNet = syncNet(name="soc_syncNet", soc=1)
#    soc_syncNet.import_model()
    soc_syncNet.hardcoded_params(params)
    soc_syncNet.build_dyad()
    nonsoc_syncNet = syncNet(name="nonsoc_syncNet", soc=0)
#    nonsoc_syncNet.import_model()
    nonsoc_syncNet.hardcoded_params(params)
    nonsoc_syncNet.build_dyad()
    return(soc_syncNet,nonsoc_syncNet)

def test_net(soc, init_val, params):
    net = syncNet(name='a', soc=soc)
    net.hardcoded_params(params,init_val)
    net.build_dyad()
    net.plug_parameters()
    net.record_interaction(time=350)
    return net

def compare_nets(net1,net2):
    return ssr(net1.dyad,net2.dyad)

def run_nets(params):
    ## social condition
    soc_syncNet = syncNet(name="soc_syncNet", file="data/socialtapping_V5-final.xlsx")
    soc_syncNet.import_model()
#    soc_syncNet.input_weights(params[0])
#    soc_syncNet.input_speed_factors(params[1])
#    soc_syncNet.input_comb_par(params[2])
    input_params(soc_syncNet, params)
    soc_syncNet.build_dyad()
    soc_syncNet.plug_parameters()
    soc_syncNet.record_interaction(time=450)
    ## nonsocial condition
    nonsoc_syncNet = syncNet(name="nonsoc_syncNet", file="data/nonsocialtapping_V5-final.xlsx")
    nonsoc_syncNet.import_model()
#    nonsoc_syncNet.input_weights(params[0])
#    nonsoc_syncNet.input_speed_factors(params[1])
#    nonsoc_syncNet.input_comb_par(params[2])
    input_params(nonsoc_syncNet, params)
    nonsoc_syncNet.build_dyad()
    nonsoc_syncNet.plug_parameters()
    nonsoc_syncNet.record_interaction(time=450)
    soc_dyad = soc_syncNet.dyad
    nonsoc_dyad = nonsoc_syncNet.dyad
    return ssr(soc_dyad, nonsoc_dyad)


    #
#  - state-wise and connection-wise condition comparisons
