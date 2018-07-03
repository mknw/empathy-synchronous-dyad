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
import random
from math import ceil

class syncNet(object):

   def __init__(self, name,soc=1):
      if soc == 1:
          self.file ="data/socialtapping_V5-final.xlsx"
      else:
          self.file ="data/nonsocialtapping_V5-final.xlsx"
      self.name = name
      self.soc = soc

   def import_model(self):
      template = pd.read_excel(self.file, [0,1])
      self.sheet1 = template[0]
      self.sheet2 = template[1]
      #states'_names
      self.sts_nms = np.asarray(self.sheet1.iloc[0:1, 0:9])[0]
#      print (self.sts_nms)
#      print (np.array(['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']))
#      print (self.sheet1.iloc[0:1, 0:9])
#      '''this function turns the sheet template into readable pandas dataframes'''
      # scrape weights into dataframe:
      weights_clean = np.asmatrix(self.sheet1.iloc[2:11, 0:9])
      weights_df = pd.DataFrame(weights_clean, index = self.sts_nms,
                                dtype="Float64")
      # label weights indices:
      weights_df.columns = self.sts_nms
      #fill NaNs with zeros
      weights_df = weights_df.fillna(0)
#      weights_df.set_value('X1','X2',10)
#      print (weights_df.get_value('X1','X2'))


      # scrape speed factors into dataframe:
      speed_factors_df = self.sheet1.iloc[13:14, 0:9]

      speed_factors_df.columns = self.sts_nms
      speed_factors_df.index = ['speed_factor']

      # scrape combination function parameters into dataframe:
      comb_par_df = self.sheet1.iloc[15:28, 0:9]
      comb_par_df.columns = self.sts_nms
#      comb_par_df.
#      comb_par_df.loc['identity function','X1']=10
#      print ('hier',comb_par_df)
      ##

      # scrape heb and hom parameters from sheet 2 into df
      adcon_par_df = self.sheet2.iloc[2:11, 1:101]
#      print (adcon_par_df)

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
      weights_df.to_pickle('weights_df.pickle%s'%self.soc)
      self.speed_factors_df = speed_factors_df
      speed_factors_df.to_pickle('speed_factors_df.pickle%s'%self.soc)
      self.comb_par_df = comb_par_df
      comb_par_df.to_pickle('comb_par_df.pickle%s'%self.soc)
      self.adcon_par_df = adcon_par_df
      adcon_par_df.to_pickle('adcon_par_df.pickle%s'%self.soc)
      self.init_states = [float(n) for n in[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
      return

   def hardcoded_params(self, params,init_val):
#       self.sts_nms = np.array(['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9'])
#       self.weights_df = pd.read_pickle('weights_df.pickle%s'%self.soc)
#       self.speed_factors_df = pd.read_pickle('speed_factors_df.pickle%s'%self.soc)
#       self.comb_par_df = pd.read_pickle('comb_par_df.pickle%s'%self.soc)
#       self.adcon_par_df = pd.read_pickle('adcon_par_df.pickle%s'%self.soc)
#       self.init_states = [float(n) for n in[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
       self.sts_nms = init_val[0]
       self.weights_df = init_val[1]
       self.speed_factors_df = init_val[2]
       self.comb_par_df = init_val[3]
       self.adcon_par_df = init_val[4]
       self.init_states = init_val[5]
       weights = self.weights_df
       speed = self.speed_factors_df
       comb = self.comb_par_df
       adcon = self.adcon_par_df
#       print (weights)
       weights.loc['X2']['X4']=params[0][0]
#       weights.loc['X3']['X4']=params[0][3]
       weights.loc['X3']['X4']=0.6
       weights.loc['X4']['X3']=params[0][1]
       weights.loc['X4']['X5']=params[0][2]
#       weights.loc['X5']['X7']=params[0][4]
       weights.loc['X5']['X7']=0.6
       weights.loc['X6']['X8']=params[0][0]
#       weights.loc['X7']['X8']=params[0][5]
       weights.loc['X7']['X8']=0.6
       weights.loc['X8']['X7']=params[0][2]
       weights.loc['X8']['X9']=params[0][3]
#       weights.loc['X9']['X3']=params[0][6]
       weights.loc['X9']['X3']=0.6
#       print (weights)
#       print (speed)
       speed.loc['speed_factor']['X2']=params[1][0]
       speed.loc['speed_factor']['X3']=params[1][1]
       speed.loc['speed_factor']['X4']=params[1][2]
       speed.loc['speed_factor']['X5']=params[1][3]
       speed.loc['speed_factor']['X6']=params[1][0]
       speed.loc['speed_factor']['X7']=params[1][1]
       speed.loc['speed_factor']['X8']=params[1][2]
       speed.loc['speed_factor']['X9']=params[1][3]
#       print (speed)
#       print(comb)
       comb.set_value(comb.index[3],comb.columns[3],params[3][0]*3)
       comb.set_value(comb.index[3],comb.columns[7],params[3][0]*3)
       comb.set_value(comb.index[9],comb.columns[2],params[3][1]*50)
       comb.set_value(comb.index[9],comb.columns[6],params[3][1]*50)
       comb.set_value(comb.index[10],comb.columns[2],params[3][2])
       comb.set_value(comb.index[10],comb.columns[6],params[3][2])
#       print (comb)
#       print (adcon)
       adcon.set_value(adcon.index[2],adcon.columns[0],params[3][0])
       adcon.set_value(adcon.index[2],adcon.columns[2],params[3][1])
       adcon.set_value(adcon.index[7],adcon.columns[1],(params[3][2]*0.95)+0.05)
       adcon.set_value(adcon.index[7],adcon.columns[3],(params[3][3]*0.95)+0.05)
       adcon.set_value(adcon.index[8],adcon.columns[1],params[3][4])
       adcon.set_value(adcon.index[8],adcon.columns[3],params[3][5])
#       print (adcon)

   def randomize_params(self):
      param_dict = dict()
      count=0
      for column in self.speed_factors_df.columns[1:]:
          self.speed_factors_df.set_value('speed_factor',column,random.random())
          param_dict [column] = self.speed_factors_df.values[0][count]
          count +=1
#      print(self.speed_factors_df)

   def input_weights(self, params):
       counter = 0
       weights = self.weights_df
       for index in weights.index[1:]:
           for column in weights.columns:
               if weights.get_value(index,column) == weights.get_value(index,column) and weights.get_value(index,column) !=0 :
                   weights.set_value(index,column,params[counter])
                   counter +=1
#       print(weights)

   def input_speed_factors(self,params):
       counter = 0
       for column in self.speed_factors_df.columns[1:]:
           self.speed_factors_df.set_value('speed_factor',column,params[counter])
           counter+=1

   def input_comb_par(self,params):
       counter = 0
       comb_par = self.comb_par_df
       params_auto = params
#       for i in range(4):
#           params_auto[i] = params_auto[i]*10
       for i in comb_par.index:
           for j in comb_par.columns:
               if comb_par.get_value(i,j) != 1:
                   if comb_par.get_value(i,j) == comb_par.get_value(i,j):
                       comb_par.set_value(i,j,params_auto[counter])
                       counter +=1
#       print (comb_par)

   def input_adcon_par(self,params):
       counter = 0
       adcon = self.adcon_par_df
#       print (adcon)
       for i in adcon.index[2:]:
           for j in adcon.columns:
               if adcon.get_value(i,j) != 1:
                   if adcon.get_value(i,j) == adcon.get_value(i,j):
                       adcon.set_value(i,j,ceil(params[counter]*10)/10)
                       counter +=1
#       print (adcon)

   def build_dyad(self):
      '''method to shape the graph's main structure and
      makes each edge and vertex ready for the 2 edges_/states_update functions'''
      dyad = nx.from_numpy_matrix(self.weights_df.values, create_using=nx.MultiDiGraph())
#      print(dyad.nodes)
      old_new = dict(zip(dyad.nodes(), self.sts_nms))
      nx.relabel_nodes(dyad, old_new, False)

      #timeline creation
      stt_d = {} #create dictionary for the 'state' attribute to be used by update functions
      stt_tmln_d = {} #create timelines dictionary for state activities

      for i in range(len(self.sts_nms)):
         #match states' names with initial weights
         stt_d[self.sts_nms[i]] = self.init_states[i]
         # match time 0 with initial states activity; assign it to states' names
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
            if adcon_par_df.loc[r][c] == adcon_par_df.loc[r][c]: # check if not NaN

               value = adcon_par_df.loc[r][c]
#               print("nodes: {}, Par:{}, Value:{}".format(c, r, value))

               # find edge by indexing the dataframe column (tuple), update its value:
               set_edge_attr = "self.dyad[c[0]][c[1]][0].update({}=value)".format(r)
               exec(set_edge_attr)
         # a bit of feedback:
#         print('Edge for nodes: {} saved to nx graph'.format(c))

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
#         if uptd_nds:
#            print("\"{}\" attribute updated for nodes: {} in nx graph".format(r, uptd_nds))

      ########## plug speed factors in states ############
      speed_factors_df = self.speed_factors_df
      spf_d = {}
      for stt in speed_factors_df.columns:
         spf_d[stt] = speed_factors_df.loc['speed_factor'][stt]
      nx.set_node_attributes(self.dyad, spf_d, "speed_factor")
#      print('Speed factors for all nodes saved to nx graph')
      return


   def record_interaction(self, time=100, delta=0.2):
      dyad = self.dyad

      for t in range(1, time): # OC original code
         temp_dyad = edges_update(dyad, t, delta)
         dyad = states_update(temp_dyad, t, delta)

#      for t in range(1, time): # update only states
#         dyad = states_update(dyad, t, delta)
#      print("Adaptive Timeline created for: " + str(self.name))
      self.dyad = dyad # networkx graph
      return

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

   def get_states_dict(self):
       real_names = [r"$ws_s$",
                     r"$srs_{A, s}$", r"$srs_{A, esB}$", r"$ps_{A, a}$", r"$es_{A, a}$",
                     r"$srs_{B, s}$", r"$srs_{B, esA}$", r"$ps_{B, a}$", r"$es_{B, a}$"]
       real_names_dict = dict(zip(self.sts_nms, real_names))
       return real_names_dict

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
    world = 1
    init_val0 = [
        np.array(['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']),
        pd.read_pickle('weights_df.pickle0'),
        pd.read_pickle('speed_factors_df.pickle0'),
        pd.read_pickle('comb_par_df.pickle0'),
        pd.read_pickle('adcon_par_df.pickle0'),
        [float(n) for n in[world, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]
    init_val1 = [
        np.array(['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']),
        pd.read_pickle('weights_df.pickle1'),
        pd.read_pickle('speed_factors_df.pickle1'),
        pd.read_pickle('comb_par_df.pickle1'),
        pd.read_pickle('adcon_par_df.pickle1'),

        [float(n) for n in[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]

    sync_dyad = syncNet('dyad',0) # CHANGE THIS TO PLOT THE RIGHT ONE
#   sync_dyad.import_model()
    params = """
       0.06228467, 0.48908727, 0.20931008, 0.35909979, 0.33157945,
       0.99409521, 0.33278341, 0.00135056, 0.53731928, 0.20758233,
       0.31189825, 0.3395705 , 0.55675687, 0.5055475 , 0.00100012,
       0.72745712, 0.56565876, 0.35077148, 0.77661382, 0.37891303
    """

    params = [float(n) for n in params.split(',')]# -*- coding: utf-8 -*-
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
import random
from math import ceil

class syncNet(object):

   def __init__(self, name,soc=1):
      if soc == 1:
          self.file ="data/socialtapping_V5-final.xlsx"
      else:
          self.file ="data/nonsocialtapping_V5-final.xlsx"
      self.name = name
      self.soc = soc

   def import_model(self):
      template = pd.read_excel(self.file, [0,1])
      self.sheet1 = template[0]
      self.sheet2 = template[1]
      #states'_names
      self.sts_nms = np.asarray(self.sheet1.iloc[0:1, 0:9])[0]
#      print (self.sts_nms)
#      print (np.array(['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']))
#      print (self.sheet1.iloc[0:1, 0:9])
#      '''this function turns the sheet template into readable pandas dataframes'''
      # scrape weights into dataframe:
      weights_clean = np.asmatrix(self.sheet1.iloc[2:11, 0:9])
      weights_df = pd.DataFrame(weights_clean, index = self.sts_nms,
                                dtype="Float64")
      # label weights indices:
      weights_df.columns = self.sts_nms
      #fill NaNs with zeros
      weights_df = weights_df.fillna(0)
#      weights_df.set_value('X1','X2',10)
#      print (weights_df.get_value('X1','X2'))


      # scrape speed factors into dataframe:
      speed_factors_df = self.sheet1.iloc[13:14, 0:9]

      speed_factors_df.columns = self.sts_nms
      speed_factors_df.index = ['speed_factor']

      # scrape combination function parameters into dataframe:
      comb_par_df = self.sheet1.iloc[15:28, 0:9]
      comb_par_df.columns = self.sts_nms
#      comb_par_df.
#      comb_par_df.loc['identity function','X1']=10
#      print ('hier',comb_par_df)
      ##

      # scrape heb and hom parameters from sheet 2 into df
      adcon_par_df = self.sheet2.iloc[2:11, 1:101]
#      print (adcon_par_df)

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
      weights_df.to_pickle('weights_df.pickle%s'%self.soc)
      self.speed_factors_df = speed_factors_df
      speed_factors_df.to_pickle('speed_factors_df.pickle%s'%self.soc)
      self.comb_par_df = comb_par_df
      comb_par_df.to_pickle('comb_par_df.pickle%s'%self.soc)
      self.adcon_par_df = adcon_par_df
      adcon_par_df.to_pickle('adcon_par_df.pickle%s'%self.soc)
      self.init_states = [float(n) for n in[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
      return

   def hardcoded_params(self, params,init_val):
#       self.sts_nms = np.array(['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9'])
#       self.weights_df = pd.read_pickle('weights_df.pickle%s'%self.soc)
#       self.speed_factors_df = pd.read_pickle('speed_factors_df.pickle%s'%self.soc)
#       self.comb_par_df = pd.read_pickle('comb_par_df.pickle%s'%self.soc)
#       self.adcon_par_df = pd.read_pickle('adcon_par_df.pickle%s'%self.soc)
#       self.init_states = [float(n) for n in[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
       self.sts_nms = init_val[0]
       self.weights_df = init_val[1]
       self.speed_factors_df = init_val[2]
       self.comb_par_df = init_val[3]
       self.adcon_par_df = init_val[4]
       self.init_states = init_val[5]
       weights = self.weights_df
       speed = self.speed_factors_df
       comb = self.comb_par_df
       adcon = self.adcon_par_df
#       print (weights)
       weights.loc['X2']['X4']=params[0][0]
#       weights.loc['X3']['X4']=params[0][3]
       weights.loc['X3']['X4']=0.6
       weights.loc['X4']['X3']=params[0][1]
       weights.loc['X4']['X5']=params[0][2]
#       weights.loc['X5']['X7']=params[0][4]
       weights.loc['X5']['X7']=0.6
       weights.loc['X6']['X8']=params[0][0]
#       weights.loc['X7']['X8']=params[0][5]
       weights.loc['X7']['X8']=0.6
       weights.loc['X8']['X7']=params[0][2]
       weights.loc['X8']['X9']=params[0][3]
#       weights.loc['X9']['X3']=params[0][6]
       weights.loc['X9']['X3']=0.6
#       print (weights)
#       print (speed)
       speed.loc['speed_factor']['X2']=params[1][0]
       speed.loc['speed_factor']['X3']=params[1][1]
       speed.loc['speed_factor']['X4']=params[1][2]
       speed.loc['speed_factor']['X5']=params[1][3]
       speed.loc['speed_factor']['X6']=params[1][0]
       speed.loc['speed_factor']['X7']=params[1][1]
       speed.loc['speed_factor']['X8']=params[1][2]
       speed.loc['speed_factor']['X9']=params[1][3]
#       print (speed)
#       print(comb)
       comb.set_value(comb.index[3],comb.columns[3],params[3][0]*3)
       comb.set_value(comb.index[3],comb.columns[7],params[3][0]*3)
       comb.set_value(comb.index[9],comb.columns[2],params[3][1]*50)
       comb.set_value(comb.index[9],comb.columns[6],params[3][1]*50)
       comb.set_value(comb.index[10],comb.columns[2],params[3][2])
       comb.set_value(comb.index[10],comb.columns[6],params[3][2])
#       print (comb)
#       print (adcon)
       adcon.set_value(adcon.index[2],adcon.columns[0],params[3][0])
       adcon.set_value(adcon.index[2],adcon.columns[2],params[3][1])
       adcon.set_value(adcon.index[7],adcon.columns[1],(params[3][2]*0.95)+0.05)
       adcon.set_value(adcon.index[7],adcon.columns[3],(params[3][3]*0.95)+0.05)
       adcon.set_value(adcon.index[8],adcon.columns[1],params[3][4])
       adcon.set_value(adcon.index[8],adcon.columns[3],params[3][5])
#       print (adcon)

   def randomize_params(self):
      param_dict = dict()
      count=0
      for column in self.speed_factors_df.columns[1:]:
          self.speed_factors_df.set_value('speed_factor',column,random.random())
          param_dict [column] = self.speed_factors_df.values[0][count]
          count +=1
#      print(self.speed_factors_df)

   def input_weights(self, params):
       counter = 0
       weights = self.weights_df
       for index in weights.index[1:]:
           for column in weights.columns:
               if weights.get_value(index,column) == weights.get_value(index,column) and weights.get_value(index,column) !=0 :
                   weights.set_value(index,column,params[counter])
                   counter +=1
#       print(weights)

   def input_speed_factors(self,params):
       counter = 0
       for column in self.speed_factors_df.columns[1:]:
           self.speed_factors_df.set_value('speed_factor',column,params[counter])
           counter+=1

   def input_comb_par(self,params):
       counter = 0
       comb_par = self.comb_par_df
       params_auto = params
#       for i in range(4):
#           params_auto[i] = params_auto[i]*10
       for i in comb_par.index:
           for j in comb_par.columns:
               if comb_par.get_value(i,j) != 1:
                   if comb_par.get_value(i,j) == comb_par.get_value(i,j):
                       comb_par.set_value(i,j,params_auto[counter])
                       counter +=1
#       print (comb_par)

   def input_adcon_par(self,params):
       counter = 0
       adcon = self.adcon_par_df
#       print (adcon)
       for i in adcon.index[2:]:
           for j in adcon.columns:
               if adcon.get_value(i,j) != 1:
                   if adcon.get_value(i,j) == adcon.get_value(i,j):
                       adcon.set_value(i,j,ceil(params[counter]*10)/10)
                       counter +=1
#       print (adcon)

   def build_dyad(self):
      '''method to shape the graph's main structure and
      makes each edge and vertex ready for the 2 edges_/states_update functions'''
      dyad = nx.from_numpy_matrix(self.weights_df.values, create_using=nx.MultiDiGraph())
#      print(dyad.nodes)
      old_new = dict(zip(dyad.nodes(), self.sts_nms))
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
#         print('Edge for nodes: {} saved to nx graph'.format(c))

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
#         if uptd_nds:
#            print("\"{}\" attribute updated for nodes: {} in nx graph".format(r, uptd_nds))

      ########## plug speed factors in states ############
      speed_factors_df = self.speed_factors_df
      spf_d = {}
      for stt in speed_factors_df.columns:
         spf_d[stt] = speed_factors_df.loc['speed_factor'][stt]
      nx.set_node_attributes(self.dyad, spf_d, "speed_factor")
#      print('Speed factors for all nodes saved to nx graph')
      return


   def record_interaction(self, time=100, delta=0.2):
      dyad = self.dyad

      for t in range(1, time): # OC original code
         temp_dyad = edges_update(dyad, t, delta)
         dyad = states_update(temp_dyad, t, delta)

#      for t in range(1, time): # update only states
#         dyad = states_update(dyad, t, delta)
#      print("Adaptive Timeline created for: " + str(self.name))
      self.dyad = dyad # networkx graph
      return

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

   def get_states_dict(self):
       real_names = [r"$ws_s$",
                     r"$srs_{A, s}$", r"$srs_{A, esB}$", r"$ps_{A, a}$", r"$es_{A, a}$",
                     r"$srs_{B, s}$", r"$srs_{B, esA}$", r"$ps_{B, a}$", r"$es_{B, a}$"]
       real_names_dict = dict(zip(self.sts_nms, real_names))
       return real_names_dict

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
    world = 1
    init_val0 = [
        np.array(['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']),
        pd.read_pickle('weights_df.pickle0'),
        pd.read_pickle('speed_factors_df.pickle0'),
        pd.read_pickle('comb_par_df.pickle0'),
        pd.read_pickle('adcon_par_df.pickle0'),
        [float(n) for n in[world, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]
    init_val1 = [
        np.array(['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9']),
        pd.read_pickle('weights_df.pickle1'),
        pd.read_pickle('speed_factors_df.pickle1'),
        pd.read_pickle('comb_par_df.pickle1'),
        pd.read_pickle('adcon_par_df.pickle1'),

        [float(n) for n in[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]

    sync_dyad = syncNet('dyad',0) # CHANGE THIS TO PLOT THE RIGHT ONE
#   sync_dyad.import_model()
    params = """
       0.06228467, 0.48908727, 0.20931008, 0.35909979, 0.33157945,
       0.99409521, 0.33278341, 0.00135056, 0.53731928, 0.20758233,
       0.31189825, 0.3395705 , 0.55675687, 0.5055475 , 0.00100012,
       0.72745712, 0.56565876, 0.35077148, 0.77661382, 0.37891303
"""

    params = [float(n) for n in params.split(',')]
    wp = params[:7]
    sp = params[7:11]
    cp = params[11:14]
    ap = params[14:]
    formatted_params =[wp,sp,cp,ap]
    print (formatted_params)

#    sync_dyad.import_model()
    sync_dyad.hardcoded_params(formatted_params,init_val0)
    sync_dyad.build_dyad()
    sync_dyad.plug_parameters()
    sync_dyad.record_interaction(time=300, delta=0.2)
    sync_dyad.plot_activation()
    sync_dyad.plot_weights()


#'''testing functions'''
##### NODES COLLECTION
#sync_dyad.dyad.nodes()
##
### EDGES COLLECTION
#sync_dyad.dyad.edges()
#
## FOR EACH NODE print NAME + NX NODE attribute
#for vrtx in sync_dyad.dyad.nodes():
#   print(vrtx)
#   print(sync_dyad.dyad.node[vrtx])
#
##
#### for each EDGE print NAME, SUCCESSOR + EDGE attribute
##print(list(sync_dyad.dyad.predecessors('X1')))
#for vrtx in sync_dyad.dyad.nodes():
#   for succ in sync_dyad.dyad.successors(vrtx):
#      print(vrtx, succ)
#      print(sync_dyad.dyad[vrtx][succ])
#
#
### get specific NODE
#sync_dyad.dyad.node['X8']
#
### get specific EDGE
#sync_dyad.dyad.get_edge_data('X2', 'X4')[0]
#
##
#
#sync_dyad.dyad['X3']['X4'][0]['weightTimeLine']
#
### print NAME + edgeFUNCTION
#g = sync_dyad.dyad
#for e in g.edges():
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
### UPDATE function, works only when assigning sync_dyad.dyad to a var (g)
#vrtx = 'X7'
#t = 30
#actual_state = 1
#g.node[vrtx]['activityTimeLine'].update({t: actual_state})
#g.node['X7']['activityTimeLine'][30] = 0.5
#g.get_edge_data(source_node,target_node)[0]['weight'] = np.asscalar(new_weight)
####################################

    wp = params[:7]
    sp = params[7:11]
    cp = params[11:14]
    ap = params[14:]
    formatted_params =[wp,sp,cp,ap]
    print (formatted_params)

#    sync_dyad.import_model()
    sync_dyad.hardcoded_params(formatted_params,init_val0)
    sync_dyad.build_dyad()
    sync_dyad.plug_parameters()
    sync_dyad.record_interaction(time=300, delta=0.2)
    sync_dyad.plot_activation()
    sync_dyad.plot_weights()
    import ipdb; ipdb.set_trace()


#'''testing functions'''
##### NODES COLLECTION
#sync_dyad.dyad.nodes()
##
### EDGES COLLECTION
#sync_dyad.dyad.edges()
#
## FOR EACH NODE print NAME + NX NODE attribute
#for vrtx in sync_dyad.dyad.nodes():
#   print(vrtx)
#   print(sync_dyad.dyad.node[vrtx])
#
##
#### for each EDGE print NAME, SUCCESSOR + EDGE attribute
##print(list(sync_dyad.dyad.predecessors('X1')))
#for vrtx in sync_dyad.dyad.nodes():
#   for succ in sync_dyad.dyad.successors(vrtx):
#      print(vrtx, succ)
#      print(sync_dyad.dyad[vrtx][succ])
#
#
### get specific NODE
#sync_dyad.dyad.node['X8']
#
### get specific EDGE
#sync_dyad.dyad.get_edge_data('X2', 'X4')[0]
#
##
#
#sync_dyad.dyad['X3']['X4'][0]['weightTimeLine']
#
### print NAME + edgeFUNCTION
#g = sync_dyad.dyad
#for e in g.edges():
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
### UPDATE function, works only when assigning sync_dyad.dyad to a var (g)
#vrtx = 'X7'
#t = 30
#actual_state = 1
#g.node[vrtx]['activityTimeLine'].update({t: actual_state})
#g.node['X7']['activityTimeLine'][30] = 0.5
#g.get_edge_data(source_node,target_node)[0]['weight'] = np.asscalar(new_weight)
####################################
