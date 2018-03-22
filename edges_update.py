'''
Calculates the next time step for the state of the nodes
'''
import numpy as np


'''
The attribute state in every node defines the last state of the node. At the dictionary activityTimeLine we have the
evolution of the values for the state over time.
'''
def edges_update(graph, t, delta = 0.2):
   
    # set dictionary keys to a meaningful output relative to
    # the time input (Xpressd in secs)
#    t = round(t*delta, 1)
    
#    if t % 2 == 0:
#        print('Time: ', t)

    g = graph.copy()
#    not_changed = []

    for edge in g.edges():
        source_node, target_node = edge
        old_weight = g.get_edge_data(source_node,target_node)[0]['weight']
        source = g.node[source_node]['state']
        target = g.node[target_node]['state']
        
        #calculate weight according to attributes extracted from nx graph
        #speed_factor is different from states' speed_factor, as it's an attribute
        #  of the edge and not of the state
        if 'hebbian' in g[source_node][target_node][0]:
#            print('hebbian:' + source_node + target_node)
#            function = 'hebbian' # unnecessary
            speed_factor = g[source_node][target_node][0]['speed_factor'] 
            persistence = g[source_node][target_node][0]['persistence']

            variation = speed_factor * (source * target * (1 - target) + persistence * target)
            # temporarily changed to:
            #variation = 0.5
#            variation = speed_factor * (source * target - target**2 * old_weight) * persistence
            

        elif 'slhom' in g[source_node][target_node][0]:
#            print('slhomophily:' + source_node + target_node)
#            function = 'slhom'  # unnecessary
            speed_factor = g[source_node][target_node][0]['speed_factor']
            thres_h = g[source_node][target_node][0]['thres_t']
            amplification = g[source_node][target_node][0]['amplification']
            variation = old_weight + amplification * old_weight * (1 - old_weight) * (thres_h - np.abs(source - target))
            
        else: #check if function str is falsy
            # feeds old_weight instead of new_weight to the weightTimeLine
            g[source_node][target_node][0]['weightTimeLine'].update({t:old_weight})
            g[source_node][target_node][0]['weight'] = old_weight
#            print("no changes to weight for " + str(edge))
            continue


#        elif function == 'advanced_linear':
#            if amplification == None:
#                print('Error! Amplification not set!')
#                exit(0)
#
#            variation = (old_weight + amplification * ((1 - old_weight) * (np.abs(thres_h - np.abs(source - target))+(
#                thres_h - np.abs(source - target))) / 2 + old_weight * (np.abs(thres_h- np.abs(source - target))-(
#                thres_h - np.abs(source - target))) / 2))
#
#        elif function == 'simple_quadratic':
#            if amplification == None:
#                print('Error! Amplification not set!')
#                exit(0)
#
#            variation = (old_weight + amplification * old_weight * (1 - old_weight) * (thres_h^ 2 - np.abs(source - target) ^ 2))
#
#        elif function == 'advanced_quadratic':
#            if amplification == None:
#                print('Error! Amplification not set!')
#                exit(0)
#
#            variation = (old_weight + amplification * ((1 - old_weight) * (np.abs(thres_h^2 - np.abs(source - target)^2) + (
#            thres_h ^ 2 - np.abs(source - target) ^ 2)) / 2 + old_weight * (np.abs(thres_h ^ 2 - np.abs(source - target) ^ 2) - (
#                                                           thres_h ^ 2 - np.abs(source - target) ^ 2)) / 2))

#        else:
#            
#            exit(0)

        new_weight = old_weight + speed_factor * (variation - old_weight )*delta # homophily model

        #new_weight = new_weight if new_weight >= 0 else 0
        #new_weight = new_weight if new_weight <= 1 else 1

        try:
            
            g[source_node][target_node][0]['weightTimeLine'].update({t:new_weight})
        except:
            print(t, source, target, g[source][target]['weightTimeLine'])
            
        try:
            g[source_node][target_node][0]['weight'] = np.asscalar(new_weight)
        except:
            g[source_node][target_node][0]['weight'] = new_weight
            
    return g