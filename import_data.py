# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 14:38:07 2018

@author: Mike
"""


import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from states_update import states_update
from edges_update import edges_update


class syncNet(object):

   def __init__(self, name, file='data/socialtapping_V4-final.xlsx'):
      self.name = name
      #read excel
      template = pd.read_excel(file, [0,1])
      self.sheet1 = template[0]
      self.sheet2 = template[1]
      #states'_names
      self.sts_nms = np.asarray(self.sheet1.iloc[0:1, 0:9])[0]

   def import_model(self):
      '''this function turns the sheet template into readable pandas dataframes'''

      # scrape weights into dataframe:
      weights_clean = np.asmatrix(self.sheet1.iloc[2:11, 0:9])
      weights_df = pd.DataFrame(weights_clean, index = self.sts_nms,
                                dtype="Float64")
      # label weights indices:
      weights_df.columns = self.sts_nms
      #fill NaNs with zeros
      weights_df = weights_df.fillna(0)


      # scrape speed factors into dataframe:
      speed_factors_df = self.sheet1.iloc[13:14, 0:9]
      speed_factors_df.columns = self.sts_nms
      speed_factors_df.index = ['speed_factor']
      ##

      # scrape combination function parameters into dataframe:
      comb_par_df = self.sheet1.iloc[15:28, 0:9]
      comb_par_df.columns = self.sts_nms
      ##

      # scrape heb and hom parameters from sheet 2 into df
      adcon_par_df = self.sheet2.iloc[2:11, 1:101]

      # scrape parameter names...
      par_ar = np.asarray(self.sheet2.iloc[2:11, 0:1])
      #...and assign them to parameter indices
      par_nms = []
      for pn in par_ar: #remove unwanted tuples caused by comas
         par_nms.append(''.join(pn[0]))
      adcon_par_df.index = par_nms

      # create state tuples, e.g. ('X1', 'x1') ... ('X10', 'x10')
      tot_sts_nms = np.append(self.sts_nms, 'X10')
      orig_sts = [stt for stt in tot_sts_nms for y in range(10)]
      dest_sts = [stt for stt in tot_sts_nms] * 10
      sts_tpls = zip(orig_sts, dest_sts)
      # finally, assign cols to hebbian and homophily parameters
      adcon_par_df.columns = sts_tpls

      # and drop unused columns
      adcon_par_df = adcon_par_df.dropna(1, 'all')
      ##

      #assign df to syncNet
      self.weights_df = weights_df
      self.speed_factors_df = speed_factors_df
      self.comb_par_df = comb_par_df
      self.adcon_par_df = adcon_par_df
      self.init_states = [float(n) for n in[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
      return

   def build_dyad(self):
      '''method to shape the graph's main structure and
      makes each edge and vertex ready for the 2 edges_/states_update functions'''
      dyad = nx.from_numpy_matrix(self.weights_df.values, create_using=nx.MultiDiGraph())
      old_new = dict(zip(dyad.nodes, self.sts_nms))
      nx.relabel_nodes(dyad, old_new, False)

      #timeline creation
      stt_d = {} #create dictionary for the 'state' attribute to be used by update functions
      stt_tmln_d = {} #create timelines dictionary for state activities

      for i in range(len(self.sts_nms)):
         #match states' names with initial weights
         stt_d[self.sts_nms[i]] = self.init_states[i]
         # match time 0 with initial weight; assign it to states' names
         d = {0: self.init_states[i]} # time = 0 for all dicts
         stt_tmln_d[self.sts_nms[i]] = d


      # push into graph
      nx.set_node_attributes(dyad, stt_tmln_d, 'activityTimeLine')
      nx.set_node_attributes(dyad, stt_d, 'state')

#      for x in range(len(sts_mns)):
#         for y in range(len(sts_mns)):
#            weight = (sync_dyad.weights_df[sts_mns[y]][sts_mns[x]])


      for orig, dest in dyad.edges():
         weight = dyad.get_edge_data(orig, dest)[0]['weight']
         dyad[orig][dest][0].update(weightTimeLine={0:weight})

      self.dyad = dyad
      return

   def plug_parameters(self):
      '''iterate over pandas dataframes and plug respective values into
      networkx graph. Added attributes are to be used by the eu & su functions'''
      #list of edge_update parameters:
      eu_par = ['speed_factor', 'hebbian', 'persistence', 'slhom', 'alhom',
                'sqhom', 'aqhom', 'thres_t', 'amplification']
      #list of state_update parameters: '''!no advanced adaptive logistic f. here!'''
      su_par = ['id', 'sum', 'ssum', 'scaling_factor', 'norsum', 'normalizing_factor',
                'adnorsum', 'slogistic', 'alogistic', 'steepness', 'threshold', 'adalogistic', 'PLACEHOLDER']


      ######### update EDGES attributes ##############
      adcon_par_df = self.adcon_par_df
      adcon_par_df.index = eu_par # change levels to the ones used as function arguments

      for c in adcon_par_df.columns:
         for r in adcon_par_df.index:
            if adcon_par_df.loc[r][c] == adcon_par_df.loc[r][c]: # =if not NaN

               value = adcon_par_df.loc[r][c]
#               print("nodes: {}, Par:{}, Value:{}".format(c, r, value))

               # find edge by indexing the dataframe column (tuple), update its value:
               set_edge_attr = "self.dyad[c[0]][c[1]][0].update({}=value)".format(r)
               exec(set_edge_attr)
         # a bit of feedback:
         print('Edge for nodes: {} saved to nx graph'.format(c))

      ########### update VERTICES attributes ###############
      comb_par_df = self.comb_par_df
      comb_par_df.index = su_par # change levels to the ones used as function arguments

      for r in comb_par_df.index:
         comb_d = {}
         uptd_nds = []
         for c in comb_par_df.columns:
            if comb_par_df.loc[r][c] == comb_par_df.loc[r][c]: # =if not NaN
               value = comb_par_df.loc[r][c] # save assign param to 'value'
               # assign the par value to 'r' key, which is the state's_name in the dict
               comb_d[c] = value
               uptd_nds.append(c)

         nx.set_node_attributes(self.dyad, comb_d, r) # 'r' is the name of the attribute
         if uptd_nds:
            print("\"{}\" attribute updated for nodes: {} in nx graph".format(r, uptd_nds))

      ########## plug speed factors in states ############
      speed_factors_df = self.speed_factors_df
      spf_d = {}
      for stt in speed_factors_df.columns:
         spf_d[stt] = speed_factors_df.loc['speed_factor'][stt]
      nx.set_node_attributes(self.dyad, spf_d, "speed_factor")
      print('Speed factors for all nodes saved to nx graph')
      return

   def record_interaction(self, time=100, delta=0.2):
      dyad = self.dyad

      for t in range(1, time): # OC original code
         temp_dyad = edges_update(dyad, t, delta)
         dyad = states_update(temp_dyad, t, delta)

#      for t in range(1, time): # update only states
#         dyad = states_update(dyad, t, delta)
      print("Adaptive Timeline created for: " + str(self.name))
      self.dyad = dyad # networkx graph
      return

   def get_states_dict(self):
       real_names = [r"$ws_s$",
                     r"$srs_{A, s}$", r"$srs_{A, esB}$", r"$ps_{A, a}$", r"$es_{A, a}$",
                     r"$srs_{B, s}$", r"$srs_{B, esA}$", r"$ps_{B, a}$", r"$es_{B, a}$"]
       real_names_dict = dict(zip(self.sts_nms, real_names))
       return real_names_dict

   def plot_activation(self):
      plt.figure(figsize=(20,15))
      plt.suptitle("Node and weight timelines for the " + self.name, fontsize = 25)
      plt.subplot(2, 1, 1)
      for node in self.dyad.nodes():
         state_tuples = self.dyad.node[node]['activityTimeLine'].items()
         plt.plot(*zip(*state_tuples))

      plt.ylabel(r"Activation level $_{(normalised)}$", fontsize=15)
      plt.xlabel("Time (seconds)", fontsize=15)
      plt.title("Node activation", {'fontsize': 20})
      legend = [r"$ws_s$",
                r"$srs_{A, s}$", r"$srs_{A, esB}$", r"$ps_{A, a}$", r"$es_{A, a}$",
                r"$srs_{B, s}$", r"$srs_{B, esA}$", r"$ps_{B, a}$", r"$es_{B, a}$"]
      plt.legend(legend, loc=2, fontsize='x-large')
      # plt.show()
      print("Plotted vertices:")
      print(self.sts_nms)
      return


   def plot_weights(self):
      plt.subplot(2, 1, 2)
      rl_adcon_list = []
      names_dict = self.get_states_dict()
      for edge in self.dyad.edges():
         source, target = edge
         #Select only adaptive edges based on number of attr assigned:
         # if len(self.dyad.get_edge_data(source, target)[0]) > 3:
         if len(self.dyad[source][target][0]) > 4:
            #get items:
            state_tuples = self.dyad.get_edge_data(source, target)[0]['weightTimeLine'].items()
            #unpack them:
            plt.plot(*zip(*state_tuples))

            # create legend through node names dictionary
            real_edge = str(names_dict[source] + ", " + names_dict[target])
            rl_adcon_list.append(real_edge)


      plt.ylabel(r"Weights $_{(normalised)}$", fontsize=15)
      plt.xlabel("Time (seconds)", fontsize=15)
      plt.title("Edges weights", {'fontsize': 20})
      plt.legend(rl_adcon_list, loc=3, fontsize='x-large')
      plt.tight_layout(pad=6)

      plt.show()
      print("Plotted edges:" + str(rl_adcon_list))
      return


if __name__ == '__main__':
   sync_dyad = syncNet('dyad')
   sync_dyad.import_model()
   sync_dyad.build_dyad()
   sync_dyad.plug_parameters()
   sync_dyad.record_interaction(time=30, delta=0.2)
   sync_dyad.plot_activation()
   sync_dyad.plot_weights()

#
# '''testing functions'''
# ### NODES COLLECTION
# sync_dyad.dyad.nodes()
# #
# ## EDGES COLLECTION
# sync_dyad.dyad.edges()
#
# # FOR EACH NODE print NAME + NX NODE attribute
# for vrtx in sync_dyad.dyad.nodes():
#   print(vrtx)
#   print(sync_dyad.dyad.node[vrtx])
#
# #
# ### for each EDGE print NAME, SUCCESSOR + EDGE attribute
# #print(list(sync_dyad.dyad.predecessors('X1')))
# for vrtx in sync_dyad.dyad.nodes():
#   for succ in sync_dyad.dyad.successors(vrtx):
#      print(vrtx, succ)
#      print(sync_dyad.dyad[vrtx][succ])
#
#
# ## get specific NODE
# sync_dyad.dyad.node['X8']
#
# ## get specific EDGE
# sync_dyad.dyad.get_edge_data('X2', 'X4')[0]
#
# #
#
# sync_dyad.dyad['X3']['X4'][0]['weightTimeLine']
#
# ## print NAME + edgeFUNCTION
# g = sync_dyad.dyad
# for e in g.edges():
#   orig, dest = e
#   print("from " + orig + " to " + dest)
#
#   if 'hebbian' in g[orig][dest][0]:
#      print("hebbian")
#   elif 'slhom' in g[orig][dest][0]:
#      print("slhom")
#   else:
#      print("no change in weight")
#
# ## UPDATE function, works only when assigning sync_dyad.dyad to a var (g)
# vrtx = 'X7'
# t = 30
# actual_state = 1
# g.node[vrtx]['activityTimeLine'].update({t: actual_state})
# g.node['X7']['activityTimeLine'][30] = 0.5
# g.get_edge_data(source_node,target_node)[0]['weight'] = np.asscalar(new_weight)
<<<<<<< HEAD
# ##################################
=======
# ##################################
>>>>>>> 91b649c2f13e977175057a0d3e11bcdce3b13082
