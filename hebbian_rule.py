import edges_update
from import_data import *
from states_update import *
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
# create network
G = nx.DiGraph()
G.add_edge('X3', 'X4')
G.add_edge('X4', 'X3')
# set node attributes
nx.set_node_attributes(G, {'X3': {0 : 0.5}, 'X4': {0 : 0.5}}, 'activityTimeLine')
nx.set_node_attributes(G, {'X3': 0.5,'X4': 0.5}, 'state')
nx.set_node_attributes(G, {'X3': 0.5, 'X4': 0.5}, 'speed_factor')
nx.set_node_attributes(G, {'X3': 1, 'X4': 1}, 'id')
# set edge attributes (the 0 is needed for edges_update to work)
G['X3']['X4'][0] = {'weight':0.6}
G['X3']['X4'][0].update({'weightTimeLine': { 0: 0.5 }})
G['X3']['X4'][0].update({'hebbian':1})
G['X3']['X4'][0].update({'speed_factor': 0.1})
G['X3']['X4'][0].update({'persistence': 0.95})

G['X4']['X3'][0] = {'weight':0.6}
G['X4']['X3'][0].update({'weightTimeLine': { 0: 0.6 }})
G['X4']['X3'][0].update({'hebbian':1})
G['X4']['X3'][0].update({'speed_factor': 0.1})
G['X3']['X4'][0].update({'persistence': 0.95})

import ipdb; ipdb.set_trace()

for t in range(500):
    temp_graph = edges_update(G, t, delta=0.2)
    G = states_update(temp_graph, t, delta=0.2)

plt.figure()
edge1 = G.get_edge_data('X3', 'X4')[0]['weightTimeLine'].values()
edge2 = G.get_edge_data('X4', 'X3')[0]['weightTimeLine'].values()

plt.plot((edge1, edge2))
plt.show()

import ipdb; ipdb.set_trace()

'''
# source for following hebbian algo: https://www.bonaccorso.eu/2017/08/21/ml-algorithms-addendum-hebbian-learning/
#previous function (do not uncomment)
# variation =  target * (source - old_weight * target)
######################################################
speed_factor = g[source_node][target_node][0]['speed_factor'] # consider this to be the learning rate
# recommended value for STD: s = 0.1
s = 0.1
lr = g[source_node][target_node][0]['persistence']
# recommended value for learning_rate: 0.005 < lr < 0.3
act_diff = target - source
variation = (1/(2*np.pi**2*s)*np.exp((-(act_diff/s))**2 / 2)) * lr - lr/4
new_weight = 1 / ( 1 + np.exp(-(old_weight + speed_factor * (variation)*delta))
'''
