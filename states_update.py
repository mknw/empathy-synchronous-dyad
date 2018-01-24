

'''
Calculates the next time step for the state of the nodes
'''
import numpy as np
from sys import exit


'''
The attribute state in every node defines the last state of the node. At the dictionary activityTimeLine we have the
evolution of the values for the state over time.
'''
def states_update(g, t, delta = 0.2):
    combination_functions_list = ['id', 'sum', 'ssum', 'norsum', 'adnorsum', 'slogistic', 'alogistic'] # , 'adalogistic'
    g_new = g.copy()


    # Updating the state of each node in the graph
    for vrtx in g.nodes():
        aggimpact = 0
        sum_weights = 0
        # Calculate the agregated impact from each neighbor's weight.
        for pred in g.predecessors(vrtx):
            #connect = g.get_edge_data(neigh, node)['weight']
#            weight_in = g[pred][vrtx][0]['weight'][t]
            weight_in = g.get_edge_data(pred, vrtx)[0]['weight']
            #connect = g.get_edge_data(neigh, node).values()[0]['weight']
            sum_weights = sum_weights + weight_in
            try:
                aggimpact = aggimpact + g.node[pred]['activityTimeLine'][t-1]*weight_in
            except:
                print(t, pred)
                exit
                
        #extract vertex attributes from nx graph
        speed_factor = g.node[vrtx]['speed_factor']

        # Defining aggimpact ['id', 'sum', 'ssum', 'norsum', 'adnorsum', 'slogistic', 'alogistic', 'adalogistic']
        if 'id' in g.node[vrtx] or 'sum' in g.node[vrtx]:
            aggimpact = aggimpact
        elif 'ssum' in g.node[vrtx]:
            # Use scaling_factor
            scaling_factor = g.node[vrtx]['scaling_factor']
            if scaling_factor == None:
                print('Error! Give scaling factor as an input to this function!')
            else:
                try:
                    scaling_f = scaling_factor
                    aggimpact = aggimpact/scaling_f
                except:
                    print('Scaling factor has to be a dictionary!')
                    print(scaling_factor)
                    exit(0)

        elif 'norsum' in g.node[vrtx]:
            # Use normalization_factor
            normalizing_factor = g.node[vrtx]['normalizing_factor']
            if normalizing_factor == None:
                print('Error! Give normalization factor as an input to this function!')
            else:
                try:
                    normalizing_f = normalizing_factor
                    aggimpact = aggimpact / normalizing_f
                except:
                    print('Normalization factor has to be a dictionary!')
                    print(normalizing_factor)
                    exit(0)

        elif 'adnorsum' in g.node[vrtx]:
            aggimpact = aggimpact / sum_weights

        elif 'slogistic' in g.node[vrtx]:
            steepness = g.node[vrtx]['steepness']
            threshold = g.node[vrtx]['threshold']
            if steepness == None or threshold == None:
                print('Steepness and threshold should be passed to the function for slogistic!')
                exit(0)
            try:
                steep = steepness[vrtx]
                thres = threshold[vrtx]
            except:
                print('Dictionary is not built with the right keys!')
                exit(0)
            aggimpact = 1 / (1 + np.exp(-steep * (aggimpact - thres)))

        elif 'alogistic' in g.node[vrtx]:
            steepness = g.node[vrtx]['steepness']
            threshold = g.node[vrtx]['threshold']
            if steepness == None or threshold == None:
                print('Steepness and threshold should be passed to the function for alogistic!')
                exit(0)
            try:
                steep = steepness
                thres = threshold
            except:
                print('Dictionary is not built with the right keys (alogistic)!')
                exit(0)
            aggimpact = ((1 / (1 + np.exp(-steep * (aggimpact - thres)))) - (1 / (1 + np.exp(steep * thres)))) * (1 + np.exp(-steep * thres))
            
        else:
            print('Your combination function is not in the possible list of functions:', combination_functions_list)
            exit(0)

        if aggimpact > 0:
            # new_state = store_states(i, step-1) + update_s * (aggimpact - store_states(i, step-1)); %calculate the new state value
            
            old_activity = g.node[vrtx]['state']
            new_activity = old_activity + speed_factor * (aggimpact - old_activity) * delta
            try:
                new_activity = np.asscalar(new_activity)
            except:
                new_activity=new_activity
                
            # set dictionary keys to a meaningful output relative to
            # the time input (Xpressd in secs)
            
            g_new.node[vrtx]['activityTimeLine'].update({t: new_activity}) #works
            g_new.node[vrtx]['state'] = new_activity # works
        else:
            try:
                actual_state = np.asscalar(g.node[vrtx]['state'])
            except:
                actual_state = g.node[vrtx]['state'] 
            g_new.node[vrtx]['activityTimeLine'].update({t: actual_state})
    return g_new


'''please note: ad(vanced)a(dapative)logistic function is not supported in the present version of states_update.py .
In order to use the adalogistic formula, modify the vertices attributes accordingly (or manually
fill in the parametres when calling the function) and add the following code @ l. 100 (after other elif's)'''
#        elif combination_function == 'adalogistic':
#            if steepness == None or threshold == None:
#                print('Steepness and threshold should be passed to the function for adalogistic!')
#                exit(0)
#            try:
#                steep = steepness[vrtx]
#                thres = threshold[vrtx]
#            except:
#                print('Dictionary is not built with the right keys (adalogistic)!')
#                exit(0)
#            aggimpact = ((1 / (1 + np.exp(-steep * (aggimpact - thres * sum_weights)))) - (1 / (1 + np.exp(steep * thres)))) * (1 + np.exp(-steep * thres))