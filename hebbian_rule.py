import edges_update
from import_data import *
from states_update import *
import numpy as np
import networkx as nx
# create network
G = nx.DiGraph()
G.add_edge('X3', 'X4', weight = 0.2)
# set node attributes
nx.set_node_attributes(G, {'X3': {0 : 0.1}, 'X4': {0 : 0.0}}, 'activityTimeLine')
nx.set_node_attributes(G, {'X3': 0.1,'X4': 0.0}, 'state')
nx.set_node_attributes(G, {'X3': 0.5, 'X4': 0.5}, 'speed_factor')
nx.set_node_attributes(G, {'X3': 1, 'X4': 1}, 'id')
# set edge attributes (the 0 is needed for edges_update to work)
G['X3']['X4'][0] = {'weight':0.2}
G['X3']['X4'][0].update({'weightTimeLine': { 0: 0.2 }})
G['X3']['X4'][0].update({'hebbian':1})
G['X3']['X4'][0].update({'speed_factor': 0.01})

for t in range(500):
    temp_graph = edges_update(G, t, delta=0.2)
    G = states_update(temp_graph, t, delta=0.2)

import ipdb; ipdb.set_trace()
