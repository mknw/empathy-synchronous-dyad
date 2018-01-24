# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:06:42 2018

@author: Mike
"""

import pandas as pd
import numpy as np
import networkx as nx
from states_update import states_update
from edges_update import edges_update
from import_data import syncNet


"""Create and prepare nx graphs through syncNet methods from import_data.py"""

## social condition
soc_syncNet = syncNet(file="data/socialtapping_V4-final.xlsx")

soc_syncNet.import_model()
soc_syncNet.build_dyad()
soc_syncNet.plug_parameters()
soc_syncNet.record_interaction(time=250)
soc_syncNet.plot_activation()
#soc_syncNet.plot_weights()




## nonsocial condition
nonsoc_syncNet = syncNet(file="data/nonsocialtapping_V4-final.xlsx")

nonsoc_syncNet.import_model()
nonsoc_syncNet.build_dyad()
nonsoc_syncNet.plug_parameters()
nonsoc_syncNet.record_interaction(time=250)
nonsoc_syncNet.plot_activation()
#nonsoc_syncNet.plot_weights()

#nonsoc_syncNet.dyad.nodes()
                      
                      
                      

# NOW PROCEED WITH:

#assign NX graphs created to  simplify subsequent code
soc_dyad = soc_syncNet.dyad
nonsoc_dyad = nonsoc_syncNet.dyad

#soc_dyad['X9']['X3']
#
#soc_dyad.node['X3']['activityTimeLine']
#
#soc_dyad.get_edge_data('X3', 'X4')
#
#soc_dyad.node['X3']

# =============================================================================
# #   - simulations model SSR comparison
# =============================================================================


SSR_dict = {}

for vrtx in soc_dyad.nodes(): # for each vertex;
   
   vrtx_SSR = 0
   
   social_actTL = list(soc_dyad.node[vrtx]['activityTimeLine'].values())
   
   nonsocial_actTL = list(nonsoc_dyad.node[vrtx]['activityTimeLine'].values())
   
   act_tuples = zip(social_actTL, nonsocial_actTL)
   
   for act in range(len(social_actTL)):
      vrtx_SSR += (social_actTL[act] - nonsocial_actTL[act])**2
         
   SSR_dict[vrtx] = vrtx_SSR
   
print(SSR_dict)
         
#   - state-wise and connection-wise condition comparisons
