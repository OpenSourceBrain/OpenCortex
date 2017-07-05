###############################################################
### 
### Note: OpenCortex is under active development, the API is subject to change without notice!!
### 
### Authors: Padraig Gleeson, Rokas Stanislovas
###
### This software has been funded by the Wellcome Trust, as well as a GSoC 2016 project 
### on Cortical Network develoment
###
##############################################################

import json
import math
import neuroml
import neuroml.loaders as loaders
import neuroml.writers as writers
import numpy as np
import opencortex
import operator
import os
import pyneuroml
from pyneuroml import pynml
import pyneuroml.lems
from pyneuroml.lems.LEMSSimulation import LEMSSimulation
import random
import shutil
import sys

all_cells = {}
all_included_files = []

cell_ids_vs_nml_docs = {}

to_be_copied_on_save = []

##############################################################################################

def _add_connection(projection, 
                   id, 
                   presynaptic_population, 
                   pre_cell_id, 
                   pre_seg_id, 
                   postsynaptic_population, 
                   post_cell_id, 
                   post_seg_id,
                   delay,
                   weight,
                   pre_fraction=0.5,
                   post_fraction=0.5):

    """
    Add a single connection to a projection between `presynaptic_population` and `postsynaptic_population`
    """


    opencortex.print_comment("Adding single conn %s in proj %s: %s(%s:%s:%s) -> %s(%s:%s:%s), delay: %sms, weight: %s" % (id, projection.id, \
                             presynaptic_population.id, pre_cell_id, pre_seg_id, pre_fraction, \
                             postsynaptic_population.id, post_cell_id, post_seg_id, post_fraction,
                             delay, weight))  

    connection = neuroml.ConnectionWD(id=id, \
                                      pre_cell_id="../%s/%i/%s" % (presynaptic_population.id, pre_cell_id, presynaptic_population.component), \
                                      pre_segment_id=pre_seg_id, \
                                      pre_fraction_along=pre_fraction,
                                      post_cell_id="../%s/%i/%s" % (postsynaptic_population.id, post_cell_id, postsynaptic_population.component), \
                                      post_segment_id=post_seg_id,
                                      post_fraction_along=post_fraction,
                                      delay='%s ms' % delay,
                                      weight=weight)

    projection.connection_wds.append(connection)


##############################################################################################

def add_elect_connection(projection, 
                         id, 
                         presynaptic_population, 
                         pre_cell_id, 
                         pre_seg_id, 
                         postsynaptic_population, 
                         post_cell_id, 
                         post_seg_id,
                         gap_junction_id,
                         pre_fraction=0.5,
                         post_fraction=0.5):

    """
    Add a single electrical connection (via a gap junction) to a projection between `presynaptic_population` and `postsynaptic_population`
    """

    opencortex.print_comment("Adding single electrical conn %s in proj %s: %s(%s:%s:%s) -> %s(%s:%s:%s)" % (id, projection.id, \
                             presynaptic_population.id, pre_cell_id, pre_seg_id, pre_fraction, \
                             postsynaptic_population.id, post_cell_id, post_seg_id, post_fraction))  

    connection = neuroml.ElectricalConnectionInstance(id=id, \
                                                      pre_cell="../%s/%i/%s" % (presynaptic_population.id, pre_cell_id, presynaptic_population.component), \
                                                      post_cell="../%s/%i/%s" % (postsynaptic_population.id, post_cell_id, postsynaptic_population.component), \
                                                      synapse=gap_junction_id, \
                                                      pre_segment=pre_seg_id, \
                                                      post_segment=post_seg_id, \
                                                      pre_fraction_along=pre_fraction, \
                                                      post_fraction_along=post_fraction)

    projection.electrical_connection_instances.append(connection)



##############################################################################################

def add_probabilistic_projection_list(net,
                                      presynaptic_population, 
                                      postsynaptic_population, 
                                      synapse_list, 
                                      connection_probability,
                                      delay=0,
                                      weight=1,
                                      presynaptic_population_list=True,
                                      postsynaptic_population_list=True,
                                      clipped_distributions=True,
                                      std_delay=None,
                                      std_weight=None):

    '''
    Modification of the method `add_probabilistic_projection()` to allow multiple synaptic components per physical projection;
    specifically works for networks containing single-compartment neuronal models. This method also allows gaussian variation in synaptic weight and delay;
    it also accepts populations that do not necessarily have the type attribute in <population> set to `populationList` .
    '''

    if presynaptic_population.size == 0 or postsynaptic_population.size == 0:
        return None

    proj_components = {}

    for synapse_id in synapse_list:

        proj = neuroml.Projection(id="%s_%s_%s" % (synapse_id, presynaptic_population.id, postsynaptic_population.id), 
                                  presynaptic_population=presynaptic_population.id, 
                                  postsynaptic_population=postsynaptic_population.id, 
                                  synapse=synapse_id)

        proj_components[synapse_id] = proj

    count = 0

    ######### check whether delay and weight varies with a synaptic component

    if isinstance(delay, list):

        if not len(delay) == len(synapse_list):

            opencortex.print_comment_v("Error in method opencortex.build.add_probabilistic_projection_list() : argument delay is a list but not of the same length as"
                                       " argument synapse_list; execution will terminate.")
            quit()

    if isinstance(weight, list):

        if not len(weight) == len(synapse_list):

            opencortex.print_comment_v("Error in method opencortex.build.add_probabilistic_projection_list() : argument weight is a list but not of the same length as"
                                       " argument synapse_list; execution will terminate.")
            quit()

    if std_delay != None:

        if isinstance(std_delay, list):

            if not len(std_delay) == len(synapse_list):

                opencortex.print_comment_v("Error in method opencortex.build.add_probabilistic_projection_list() : argument std_delay is a list but not of the same length as"
                                           " argument synapse_list; execution will terminate.")
                quit()

    if std_weight != None:

        if isinstance(std_weight, list):

            if not len(std_weight) == len(synapse_list):

                opencortex.print_comment_v("Error in method opencortex.build.add_probabilistic_projection_list() : argument std_weight is a list but not of the same length as"
                                           " argument synapse_list; execution will terminate.")
                quit()

    for i in range(0, presynaptic_population.size):
        for j in range(0, postsynaptic_population.size):
            if i != j or presynaptic_population.id != postsynaptic_population.id:

                if connection_probability >= 1 or random.random() < connection_probability:

                    if not isinstance(delay, list):
                        if std_delay != None:
                            if clipped_distributions:
                                found_positive_delay = False

                                while not found_positive_delay:
                                    del_val = random.gauss(delay, std_delay)
                                    if del_val >= 0:
                                        found_positive_delay = True
                            else:
                                del_val = random.gauss(delay, std_delay)
                        else:
                            del_val = delay

                    if not isinstance(weight, list):
                        if std_weight != None:
                            if clipped_distributions:
                                found_signed_weight = False
                                while not found_signed_weight:
                                    w_val = random.gauss(weight, std_weight)
                                    if weight > 0:
                                        if w_val >= 0:
                                            found_signed_weight = True

                                    elif weight < 0:
                                        if w_val <= 0:
                                            found_signed_weight = True

                                    else:
                                        found_signed_weight = True

                            else:
                                w_val = random.gauss(weight, std_weight)

                        else:
                            w_val = weight

                    syn_counter = 0

                    for synapse_id in synapse_list:

                        if isinstance(delay, list):

                            if std_delay != None:
                                if isinstance(std_delay, list):
                                    if clipped_distributions:
                                        found_positive_delay = False
                                        while not found_positive_delay:
                                            del_val = random.gauss(delay[syn_counter], std_delay[syn_counter])
                                            if del_val >= 0:
                                                found_positive_delay = True
                                    else:
                                        del_val = random.gauss(delay[syn_counter], std_delay[syn_counter])

                                else:

                                    if clipped_distributions:
                                        found_positive_delay = False
                                        while not found_positive_delay:
                                            del_val = random.gauss(delay[syn_counter], std_delay)

                                            if del_val >= 0:
                                                found_positive_delay = True
                                    else:
                                        del_val = random.gauss(delay[syn_counter], std_delay) 

                            else:

                                del_val = delay[syn_counter]

                        if isinstance(weight, list):

                            if std_weight != None:
                                if isinstance(std_weight, list):
                                    if clipped_distributions:
                                        found_signed_weight = False

                                        while not found_signed_weight:
                                            w_val = random.gauss(weight[syn_counter], std_weight[syn_counter])

                                            if weight[syn_counter] > 0:
                                                if w_val >= 0:
                                                    found_signed_weight = True

                                            elif weight[syn_counter] < 0:
                                                if w_val <= 0:
                                                    found_signed_weight = True

                                            else:
                                                found_signed_weight = True
                                    else:
                                        w_val = random.gauss(weight[syn_counter], std_weight[syn_counter])

                                else:

                                    if clipped_distributions:
                                        found_signed_weight = False
                                        while not found_signed_weight:
                                            w_val = random.gauss(weight[syn_counter], std_weight)

                                            if weight[syn_counter] > 0:
                                                if w_val >= 0:
                                                    found_signed_weight = True

                                            elif weight[syn_counter] < 0:
                                                if w_val <= 0:
                                                    found_signed_weight = True

                                            else:
                                                found_signed_weight = True

                                    else:
                                        w_val = random.gauss(weight[syn_counter], std_weight)

                            else:

                                w_val = weight[syn_counter]

                        if presynaptic_population_list:
                            pre_cell_string = "../%s/%i/%s" % (presynaptic_population.id, i, presynaptic_population.component)    

                        else:
                            pre_cell_string = "../%s[%i]" % (presynaptic_population.id, i)  

                        if postsynaptic_population_list:
                            post_cell_string = "../%s/%i/%s" % (postsynaptic_population.id, j, postsynaptic_population.component)

                        else:
                            post_cell_string = "../%s[%i]" % (postsynaptic_population.id, j)

                        connection = neuroml.ConnectionWD(id=count, \
                                                          pre_cell_id=pre_cell_string, \
                                                          pre_segment_id=0, \
                                                          pre_fraction_along=0.5,
                                                          post_cell_id=post_cell_string, \
                                                          post_segment_id=0,
                                                          post_fraction_along=0.5,
                                                          delay='%f ms' % del_val,
                                                          weight=w_val)

                        proj_components[synapse_id].connection_wds.append(connection)  

                        syn_counter += 1

                    count += 1

    return_proj_components = []

    if count != 0:

        for synapse_id in synapse_list:
            net.projections.append(proj_components[synapse_id])
            return_proj_components.append(proj_components[synapse_id])

        return return_proj_components

    else:

        return None



############################################################################################## 

def add_targeted_projection_by_dicts(net,
                        proj_array,
                        presynaptic_population,
                        postsynaptic_population,
                        targeting_mode,
                        synapse_list,
                        pre_seg_target_dict,
                        post_seg_target_dict,
                        subset_dict,
                        delays_dict=None,
                        weights_dict=None):


    '''This method adds the divergent or convergent chemical projection depending on the input argument targeting_mode. The input arguments are as follows:

    net - the network object created using libNeuroML API ( neuroml.Network() );

    proj_array - list which stores the projections of class neuroml.Projection; each projection has unique synapse component (e.g. AMPA , NMDA or GABA);
    thus for each projection the list position in the proj_array must be identical to the list position of the corresponding synapse id in the synapse_list;

    presynaptic_population - object corresponding to the presynaptic population in the network;

    postsynaptic_population - object corresponding to the postsynaptic population in the network;

    targeting_mode - a string that specifies the targeting mode: 'convergent' or 'divergent';

    synapse_list - the list of synapse ids that correspond to the individual receptor components on the physical synapse, e.g. the first element is
    the id of the AMPA synapse and the second element is the id of the NMDA synapse; these synapse components will be mapped onto the same location of the target segment;

    pre_seg_target_dict - a dictionary whose keys are the ids of presynaptic segment groups and the values are dictionaries in the format returned by make_target_dict();

    post_seg_target_dict - a dictionary whose keys are the ids of target segment groups and the values are dictionaries in the format returned by make_target_dict();

    subset_dict - a dictionary whose keys are the ids of target segment groups; interpretation of the corresponding dictionary values depends on the targeting mode:

    Case I, targeting mode = 'divergent' - the number of synaptic connections made by each presynaptic cell per given target segment group of postsynaptic cells;

    Case II, targeting mode = 'convergent' - the number of synaptic connections per target segment group per each postsynaptic cell;

    alternatively, subset_dict can be a number that specifies the total number of synaptic connections (either divergent or convergent) irrespective of target segment groups.

    delays_dict - optional dictionary that specifies the delays (in ms) for individual synapse components, e.g. {'NMDA':5.0} or {'AMPA':3.0,'NMDA':5};

    weights_dict - optional dictionary that specifies the weights (in ms) for individual synapse components, e.g. {'NMDA':1} or {'NMDA':1,'AMPA':2}.'''    

    opencortex.print_comment_v("Adding %s projection with %s conns: %s: %s -> %s, %s" % (targeting_mode, subset_dict, proj_array, presynaptic_population.id, postsynaptic_population.id, synapse_list))

    if targeting_mode == 'divergent':

        pop1_size = presynaptic_population.size

        pop1_id = presynaptic_population.id

        pop2_size = postsynaptic_population.size

        pop2_id = postsynaptic_population.size

    if targeting_mode == 'convergent':

        pop1_size = postsynaptic_population.size

        pop1_id = postsynaptic_population.id

        pop2_size = presynaptic_population.size

        pop2_id = presynaptic_population.id

    if isinstance(subset_dict, dict):

        numberConnections = {}

        for subset in subset_dict.keys():

            numberConnections[subset] = int(subset_dict[subset])

    if isinstance(subset_dict, int) or isinstance(subset_dict, float):

        numberConnections = int(subset_dict)

    count = 0

    for i in range(0, pop1_size):

        total_conns = 0

        if isinstance(subset_dict, dict):

            conn_subsets = {}

            for subset in subset_dict.keys():

                if subset_dict[subset] != numberConnections[subset]:

                    if random.random() < subset_dict[subset] - numberConnections[subset]:

                        conn_subsets[subset] = numberConnections[subset] + 1

                    else:

                        conn_subsets[subset] = numberConnections[subset]

                else:

                    conn_subsets[subset] = numberConnections[subset]

                total_conns = total_conns + conn_subsets[subset]

        if isinstance(subset_dict, float) or isinstance(subset_dict, int):

            conn_subsets = 0

            if subset_dict != numberConnections:

                if random.random() < subset_dict - numberConnections:

                    conn_subsets = numberConnections + 1

                else:

                    conn_subsets = numberConnections

            else:

                conn_subsets = numberConnections

            total_conns = total_conns + conn_subsets

        if total_conns != 0:    

            ##### allows only one pre segment group per presynaptic population e.g. distal_axon
            if pre_seg_target_dict != None and len(pre_seg_target_dict.keys()) == 1:

                pre_subset_dict = {}

                pre_subset_dict[pre_seg_target_dict.keys()[0]] = total_conns

            else:

                pre_subset_dict = None

            pop2_cell_ids = range(0, pop2_size)

            if pop1_id == pop2_id:

                pop2_cell_ids.remove(i)

            if pop2_cell_ids != []:

                if len(pop2_cell_ids) >= total_conns:
                    ##### get unique set of cells
                    pop2_cells = random.sample(pop2_cell_ids, total_conns)

                else:
                    #### any cell might appear several times
                    pop2_cells = []

                    for value in range(0, total_conns):

                        cell_id = random.sample(pop2_cell_ids, 1)

                        pop2_cells.extend(cell_id)

                post_target_seg_array, post_target_fractions = get_target_segments(post_seg_target_dict, conn_subsets)

                if pre_subset_dict != None:

                    pre_target_seg_array, pre_target_fractions = get_target_segments(pre_seg_target_dict, pre_subset_dict)

                else:

                    pre_target_seg_array = None

                    pre_target_fractions = None  

                for j in pop2_cells:

                    post_seg_id = post_target_seg_array[0]

                    del post_target_seg_array[0]

                    post_fraction_along = post_target_fractions[0]

                    del post_target_fractions[0]  

                    if pre_target_seg_array != None and pre_target_fractions != None:

                        pre_seg_id = pre_target_seg_array[0]

                        del pre_target_seg_array[0]

                        pre_fraction_along = pre_target_fractions[0]

                        del pre_target_fractions[0]

                    else:

                        pre_seg_id = 0

                        pre_fraction_along = 0.5

                    if targeting_mode == 'divergent':

                        pre_cell_id = i

                        post_cell_id = j

                    if targeting_mode == 'convergent':

                        pre_cell_id = j

                        post_cell_id = i  

                    syn_counter = 0

                    for synapse_id in synapse_list:

                        delay = 0

                        weight = 1

                        if delays_dict != None:
                            for synapseComp in delays_dict.keys():
                                if synapseComp in synapse_id:
                                    delay = delays_dict[synapseComp]

                        if weights_dict != None:
                            for synapseComp in weights_dict.keys():
                                if synapseComp in synapse_id:
                                    weight = weights_dict[synapseComp]

                        _add_connection(proj_array[syn_counter], 
                                       count, 
                                       presynaptic_population, 
                                       pre_cell_id, 
                                       pre_seg_id, 
                                       postsynaptic_population, 
                                       post_cell_id, 
                                       post_seg_id,
                                       delay=delay,
                                       weight=weight,
                                       pre_fraction=pre_fraction_along,
                                       post_fraction=post_fraction_along)


                        syn_counter += 1               

                    count += 1


    if count != 0:   

        for synapse_ind in range(0, len(synapse_list)):

            net.projections.append(proj_array[synapse_ind])

    return proj_array  



##############################################################################################

def _add_elect_projection(net,
                         proj_array,
                         presynaptic_population,
                         postsynaptic_population,
                         targeting_mode,
                         synapse_list,
                         pre_seg_target_dict,
                         post_seg_target_dict,
                         subset_dict):

    '''This method adds the divergent or convergent electrical projection depending on the input argument targeting_mode. The input arguments are as follows:

    net - the network object created using libNeuroML API ( neuroml.Network() );

    proj_array - list which stores the projections of class neuroml.ElectricalProjection; each projection has unique gap junction component;
    thus for each projection the list position in the proj_array must be identical to the list position of the corresponding gap junction id in the synapse_list;

    presynaptic_population - object corresponding to the presynaptic population in the network;

    postsynaptic_population - object corresponding to the postsynaptic population in the network;

    targeting_mode - a string that specifies the targeting mode: 'convergent' or 'divergent';

    synapse_list - the list of gap junction (synapse) ids that correspond to the individual gap junction components on the physical contact;
    these components will be mapped onto the same location of the target segment;

    pre_seg_target_dict - a dictionary whose keys are the ids of presynaptic segment groups and the values are dictionaries in the format returned by make_target_dict();

    post_seg_target_dict - a dictionary whose keys are the ids of target segment groups and the values are dictionaries in the format returned by make_target_dict();

    subset_dict - a dictionary whose keys are the ids of target segment groups; interpretation of the corresponding dictionary values depends on the targeting mode:

    Case I, targeting mode = 'divergent' - the number of synaptic connections made by each presynaptic cell per given target segment group of postsynaptic cells;

    Case II, targeting mode = 'convergent' - the number of synaptic connections per target segment group per each postsynaptic cell;

    alternatively, subset_dict can be a number that specifies the total number of synaptic connections (either divergent or convergent) irrespective of target segment groups.'''    

    if targeting_mode == 'divergent':

        pop1_size = presynaptic_population.size

        pop1_id = presynaptic_population.id

        pop2_size = postsynaptic_population.size

        pop2_id = postsynaptic_population.size

    if targeting_mode == 'convergent':

        pop1_size = postsynaptic_population.size

        pop1_id = postsynaptic_population.id

        pop2_size = presynaptic_population.size

        pop2_id = presynaptic_population.id

    count = 0

    if isinstance(subset_dict, dict):

        numberConnections = {}

        for subset in subset_dict.keys():

            numberConnections[subset] = int(subset_dict[subset])

    if isinstance(subset_dict, int) or isinstance(subset_dict, float):

        numberConnections = int(subset_dict)

    for i in range(0, pop1_size):

        total_conns = 0

        if isinstance(subset_dict, dict):

            conn_subsets = {}

            for subset in subset_dict.keys():

                if subset_dict[subset] != numberConnections[subset]:

                    if random.random() < subset_dict[subset] - numberConnections[subset]:

                        conn_subsets[subset] = numberConnections[subset] + 1

                    else:

                        conn_subsets[subset] = numberConnections[subset]

                else:

                    conn_subsets[subset] = numberConnections[subset]

                total_conns = total_conns + conn_subsets[subset]

        if isinstance(subset_dict, float) or isinstance(subset_dict, int):

            conn_subsets = 0

            if subset_dict != numberConnections:

                if random.random() < subset_dict - numberConnections:

                    conn_subsets = numberConnections + 1

                else:

                    conn_subsets = numberConnections

            else:

                conn_subsets = numberConnections

            total_conns = total_conns + conn_subsets

        if total_conns != 0:    

            if pre_seg_target_dict != None and len(pre_seg_target_dict.keys()) == 1:

                pre_subset_dict = {}

                pre_subset_dict[pre_seg_target_dict.keys()[0]] = total_conns

            else:

                pre_subset_dict = None

            pop2_cell_ids = range(0, pop2_size)

            if pop1_id == pop2_id:

                pop2_cell_ids.remove(i)

            if pop2_cell_ids != []:

                if len(pop2_cell_ids) >= total_conns:

                    pop2_cells = random.sample(pop2_cell_ids, total_conns)

                else:

                    pop2_cells = []

                    for value in range(0, total_conns):

                        cell_id = random.sample(pop2_cell_ids, 1)

                        pop2_cells.extend(cell_id)

                post_target_seg_array, post_target_fractions = get_target_segments(post_seg_target_dict, conn_subsets)

                if pre_subset_dict != None:

                    pre_target_seg_array, pre_target_fractions = get_target_segments(pre_seg_target_dict, pre_subset_dict)

                else:

                    pre_target_seg_array = None

                    pre_target_fractions = None 

                for j in pop2_cells:

                    post_seg_id = post_target_seg_array[0]

                    del post_target_seg_array[0]

                    post_fraction_along = post_target_fractions[0]

                    del post_target_fractions[0]  

                    if pre_target_seg_array != None and pre_target_fractions != None:

                        pre_seg_id = pre_target_seg_array[0]

                        del pre_target_seg_array[0]

                        pre_fraction_along = pre_target_fractions[0]

                        del pre_target_fractions[0]

                    else:

                        pre_seg_id = 0

                        pre_fraction_along = 0.5

                    if targeting_mode == 'divergent':

                        pre_cell_id = i

                        post_cell_id = j

                    if targeting_mode == 'convergent':

                        pre_cell_id = j

                        post_cell_id = i  

                    syn_counter = 0   

                    for synapse_id in synapse_list:

                        add_elect_connection(proj_array[syn_counter], 
                                             count, 
                                             presynaptic_population, 
                                             pre_cell_id, 
                                             pre_seg_id, 
                                             postsynaptic_population, 
                                             post_cell_id, 
                                             post_seg_id,
                                             synapse_id,
                                             pre_fraction=pre_fraction_along,
                                             post_fraction=post_fraction_along)

                        syn_counter += 1                      

                    count += 1

    if count != 0:

        for synapse_ind in range(0, len(synapse_list)):

            net.electrical_projections.append(proj_array[synapse_ind])

    return proj_array   


##############################################################################################

def add_chem_spatial_projection(net,
                                proj_array,
                                presynaptic_population,
                                postsynaptic_population,
                                targeting_mode,
                                synapse_list,
                                pre_seg_target_dict,
                                post_seg_target_dict,
                                subset_dict,
                                distance_rule,
                                pre_cell_positions,
                                post_cell_positions,
                                delays_dict,
                                weights_dict):


    '''This method adds the divergent distance-dependent chemical projection. The input arguments are as follows:


    net - the network object created using libNeuroML API ( neuroml.Network() );

    proj_array - list which stores the projections of class neuroml.Projection; each projection has unique synapse component (e.g. AMPA , NMDA or GABA);
    thus for each projection the list position in the proj_array must be identical to the list position of the corresponding synapse id in the synapse_list;

    presynaptic_population - object corresponding to the presynaptic population in the network;

    postsynaptic_population - object corresponding to the postsynaptic population in the network;

    targeting_mode - a string that specifies the targeting mode: 'convergent' or 'divergent';

    synapse_list - the list of synapse ids that correspond to the individual receptor components on the physical synapse, e.g. the first element is
    the id of the AMPA synapse and the second element is the id of the NMDA synapse; these synapse components will be mapped onto the same location of the target segment;

    pre_seg_target_dict - a dictionary whose keys are the ids of presynaptic segment groups and the values are dictionaries in the format returned by make_target_dict();

    post_seg_target_dict - a dictionary whose keys are the ids of target segment groups and the values are dictionaries in the format returned by make_target_dict();

    subset_dict - a dictionary whose keys are the ids of target segment groups; interpretation of the corresponding dictionary values depends on the targeting mode:

    Case I, targeting mode = 'divergent' - the desired number of synaptic connections made by each presynaptic cell per given target segment group of postsynaptic cells;

    Case II, targeting mode = 'convergent' - the desired number of synaptic connections per target segment group per each postsynaptic cell;

    alternatively, subset_dict can be a number that specifies the total number of synaptic connections (either divergent or convergent) irrespective of target segment groups.

    Note: the chemical connection is made only if distance-dependent probability is higher than some random number random.random(); thus, the actual numbers of connections made

    according to the distance-dependent rule might be smaller than the numbers of connections specified by subset_dict; subset_dict defines the upper bound for the 

    number of connections.

    distance_rule - string which defines the distance dependent rule of connectivity - soma to soma distance must be represented by the string character 'r';

    pre_cell_positions- array specifying the cell positions for the presynaptic population; the format is an array of [ x coordinate, y coordinate, z coordinate];

    post_cell_positions- array specifying the cell positions for the postsynaptic population; the format is an array of [ x coordinate, y coordinate, z coordinate];

    delays_dict - optional dictionary that specifies the delays (in ms) for individual synapse components, e.g. {'NMDA':5.0} or {'AMPA':3.0,'NMDA':5};

    weights_dict - optional dictionary that specifies the weights (in ms) for individual synapse components, e.g. {'NMDA':1} or {'NMDA':1,'AMPA':2}.'''   

    if targeting_mode == 'divergent':

        pop1_size = presynaptic_population.size

        pop1_id = presynaptic_population.id

        pop1_cell_positions = pre_cell_positions

        pop2_size = postsynaptic_population.size

        pop2_id = postsynaptic_population.size

        pop2_cell_positions = post_cell_positions

    if targeting_mode == 'convergent':

        pop1_size = postsynaptic_population.size

        pop1_id = postsynaptic_population.id

        pop1_cell_positions = post_cell_positions

        pop2_size = presynaptic_population.size

        pop2_id = presynaptic_population.id

        pop2_cell_positions = pre_cell_positions

    if isinstance(subset_dict, dict):

        numberConnections = {}

        for subset in subset_dict.keys():

            numberConnections[subset] = int(subset_dict[subset])

    if isinstance(subset_dict, int) or isinstance(subset_dict, float):

        numberConnections = int(subset_dict)

    count = 0

    for i in range(0, pop1_size):

        total_conns = 0

        if isinstance(subset_dict, dict):

            conn_subsets = {}

            for subset in subset_dict.keys():

                if subset_dict[subset] != numberConnections[subset]:

                    if random.random() < subset_dict[subset] - numberConnections[subset]:

                        conn_subsets[subset] = numberConnections[subset] + 1

                    else:

                        conn_subsets[subset] = numberConnections[subset]

                else:

                    conn_subsets[subset] = numberConnections[subset]

                total_conns = total_conns + conn_subsets[subset]

        if isinstance(subset_dict, float) or isinstance(subset_dict, int):

            conn_subsets = 0

            if subset_dict != numberConnections:

                if random.random() < subset_dict - numberConnections:

                    conn_subsets = numberConnections + 1

                else:

                    conn_subsets = numberConnections

            else:

                conn_subsets = numberConnections

            total_conns = total_conns + conn_subsets

        if total_conns != 0:    

            if pre_seg_target_dict != None and len(pre_seg_target_dict.keys()) == 1:

                pre_subset_dict = {}

                pre_subset_dict[pre_seg_target_dict.keys()[0]] = total_conns

            else:

                pre_subset_dict = None

            pop2_cell_ids = range(0, pop2_size)

            if pop1_id == pop2_id:

                pop2_cell_ids.remove(i)

            if pop2_cell_ids != []:

                cell1_position = pop1_cell_positions[i]

                post_target_seg_array, post_fractions_along = get_target_segments(post_seg_target_dict, conn_subsets)

                if pre_subset_dict != None:

                    pre_target_seg_array, pre_target_fractions = get_target_segments(pre_seg_target_dict, pre_subset_dict)

                else:

                    pre_target_seg_array = None

                    pre_target_fractions = None 

                conn_counter = 0

                for j in pop2_cell_ids:

                    cell2_position = pop2_cell_positions[j]

                    r = math.sqrt(sum([(a - b) ** 2 for a, b in zip(cell1_position, cell2_position)]))

                    if eval(distance_rule) >= 1 or random.random() < eval(distance_rule):

                        conn_counter += 1

                        post_seg_id = post_target_seg_array[0]

                        del post_target_seg_array[0]

                        post_fraction_along = post_fractions_along[0]

                        del post_fractions_along[0]

                        if pre_target_seg_array != None and pre_target_fractions != None:

                            pre_seg_id = pre_target_seg_array[0]

                            del pre_target_seg_array[0]

                            pre_fraction_along = pre_target_fractions[0]

                            del pre_target_fractions[0]

                        else:

                            pre_seg_id = 0

                            pre_fraction_along = 0.5

                        if targeting_mode == 'divergent':

                            pre_cell_id = i

                            post_cell_id = j

                        if targeting_mode == 'convergent':

                            pre_cell_id = j

                            post_cell_id = i

                        syn_counter = 0

                        for synapse_id in synapse_list:

                            delay = 0

                            weight = 1

                            if delays_dict != None:
                                for synapseComp in delays_dict.keys():
                                    if synapseComp in synapse_id:
                                        delay = delays_dict[synapseComp]

                            if weights_dict != None:
                                for synapseComp in weights_dict.keys():
                                    if synapseComp in synapse_id:
                                        weight = weights_dict[synapseComp]


                            _add_connection(proj_array[syn_counter], 
                                           count, 
                                           presynaptic_population, 
                                           pre_cell_id, 
                                           pre_seg_id, 
                                           postsynaptic_population, 
                                           post_cell_id, 
                                           post_seg_id,
                                           delay=delay,
                                           weight=weight,
                                           pre_fraction=pre_fraction_along,
                                           post_fraction=post_fraction_along)


                            syn_counter += 1

                        count += 1

                    if conn_counter == total_conns:
                        break

    if count != 0:

        for synapse_ind in range(0, len(synapse_list)):

            net.projections.append(proj_array[synapse_ind])

    return proj_array               


##############################################################################################

def add_elect_spatial_projection(net,
                                 proj_array,
                                 presynaptic_population,
                                 postsynaptic_population,
                                 targeting_mode,
                                 synapse_list,
                                 pre_seg_target_dict,
                                 post_seg_target_dict,
                                 subset_dict,
                                 distance_rule,
                                 pre_cell_positions,
                                 post_cell_positions):

    '''This method adds the divergent or convergent electrical projection depending on the input argument targeting_mode. The input arguments are as follows:

    net - the network object created using libNeuroML API ( neuroml.Network() );

    proj_array - dictionary which stores the projections of class neuroml.ElectricalProjection; each projection has unique gap junction component;
    thus for each projection the list position in the proj_array must be identical to the list position of the corresponding gap junction id in the synapse_list;

    presynaptic_population - object corresponding to the presynaptic population in the network;

    postsynaptic_population - object corresponding to the postsynaptic population in the network;

    targeting_mode - a string that specifies the targeting mode: 'convergent' or 'divergent';

    synapse_list - the list of gap junction (synapse) ids that correspond to the individual gap junction components on the physical contact;
    these components will be mapped onto the same location of the target segment;

    pre_seg_target_dict - a dictionary whose keys are the ids of presynaptic segment groups and the values are dictionaries in the format returned by make_target_dict();

    post_seg_target_dict - a dictionary whose keys are the ids of target segment groups and the values are dictionaries in the format returned by make_target_dict();

    subset_dict - a dictionary whose keys are the ids of target segment groups; interpretation of the corresponding dictionary values depends on the targeting mode:

    Case I, targeting mode = 'divergent' - the number of synaptic connections made by each presynaptic cell per given target segment group of postsynaptic cells;

    Case II, targeting mode = 'convergent' - the number of synaptic connections per target segment group per each postsynaptic cell;

    alternatively, subset_dict can be a number that specifies the total number of synaptic connections (either divergent or convergent) irrespective of target segment groups.

    Note: the electrical connection is made only if distance-dependent probability is higher than some random number random.random(); thus, the actual numbers of connections made

    according to the distance-dependent rule might be smaller than the numbers of connections specified by subset_dict; subset_dict defines the upper bound for the 

    number of connections.

    distance_rule - string which defines the distance dependent rule of connectivity - soma to soma distance must be represented by the string character 'r';

    pre_cell_positions- array specifying the cell positions for the presynaptic population; the format is an array of [ x coordinate, y coordinate, z coordinate];

    post_cell_positions- array specifying the cell positions for the postsynaptic population; the format is an array of [ x coordinate, y coordinate, z coordinate];'''    

    if targeting_mode == 'divergent':

        pop1_size = presynaptic_population.size

        pop1_id = presynaptic_population.id

        pop1_cell_positions = pre_cell_positions

        pop2_size = postsynaptic_population.size

        pop2_id = postsynaptic_population.size

        pop2_cell_positions = post_cell_positions

    if targeting_mode == 'convergent':

        pop1_size = postsynaptic_population.size

        pop1_id = postsynaptic_population.id

        pop1_cell_positions = post_cell_positions

        pop2_size = presynaptic_population.size

        pop2_id = presynaptic_population.id

        pop2_cell_positions = pre_cell_positions

    count = 0

    if isinstance(subset_dict, dict):

        numberConnections = {}

        for subset in subset_dict.keys():

            numberConnections[subset] = int(subset_dict[subset])

    if isinstance(subset_dict, int) or isinstance(subset_dict, float):

        numberConnections = int(subset_dict)

    for i in range(0, pop1_size):

        total_conns = 0

        if isinstance(subset_dict, dict):

            conn_subsets = {}

            for subset in subset_dict.keys():

                if subset_dict[subset] != numberConnections[subset]:

                    if random.random() < subset_dict[subset] - numberConnections[subset]:

                        conn_subsets[subset] = numberConnections[subset] + 1

                    else:

                        conn_subsets[subset] = numberConnections[subset]

                else:

                    conn_subsets[subset] = numberConnections[subset]

                total_conns = total_conns + conn_subsets[subset]

        if isinstance(subset_dict, float) or isinstance(subset_dict, int):

            conn_subsets = 0

            if subset_dict != numberConnections:

                if random.random() < subset_dict - numberConnections:

                    conn_subsets = numberConnections + 1

                else:

                    conn_subsets = numberConnections

            else:

                conn_subsets = numberConnections

            total_conns = total_conns + conn_subsets

        if total_conns != 0:  

            if pre_seg_target_dict != None and len(pre_seg_target_dict.keys()) == 1:

                pre_subset_dict = {}

                pre_subset_dict[pre_seg_target_dict.keys()[0]] = total_conns

            else:

                pre_subset_dict = None  

            pop2_cell_ids = range(0, pop2_size)

            if pop1_id == pop2_id:

                pop2_cell_ids.remove(i)

            if pop2_cell_ids != []:

                cell1_position = pop1_cell_positions[i]

                post_target_seg_array, post_target_fractions = get_target_segments(post_seg_target_dict, conn_subsets)

                if pre_subset_dict != None:

                    pre_target_seg_array, pre_target_fractions = get_target_segments(pre_seg_target_dict, pre_subset_dict)

                else:

                    pre_target_seg_array = None

                    pre_target_fractions = None 

                conn_counter = 0

                for j in pop2_cell_ids:

                    cell2_position = pop2_cell_positions[j]

                    r = math.sqrt(sum([(a - b) ** 2 for a, b in zip(cell1_position, cell2_position)]))

                    if eval(distance_rule) >= 1 or random.random() < eval(distance_rule):

                        conn_counter += 1

                        post_seg_id = post_target_seg_array[0]

                        del post_target_seg_array[0]

                        post_fraction_along = post_target_fractions[0]

                        del post_target_fractions[0]  

                        if pre_target_seg_array != None and pre_target_fractions != None:

                            pre_seg_id = pre_target_seg_array[0]

                            del pre_target_seg_array[0]

                            pre_fraction_along = pre_target_fractions[0]

                            del pre_target_fractions[0]

                        else:

                            pre_seg_id = 0

                            pre_fraction_along = 0.5

                        if targeting_mode == 'divergent':

                            pre_cell_id = i

                            post_cell_id = j

                        if targeting_mode == 'convergent':

                            pre_cell_id = j

                            post_cell_id = i  

                        syn_counter = 0   

                        for synapse_id in synapse_list:

                            add_elect_connection(proj_array[syn_counter], 
                                                 count, 
                                                 presynaptic_population, 
                                                 pre_cell_id, 
                                                 pre_seg_id, 
                                                 postsynaptic_population, 
                                                 post_cell_id, 
                                                 post_seg_id,
                                                 synapse_id,
                                                 pre_fraction=pre_fraction_along,
                                                 post_fraction=post_fraction_along)

                            syn_counter += 1                      

                        count += 1

                    if conn_counter == total_conns:
                        break

    if count != 0:

        for synapse_ind in range(0, len(synapse_list)):

            net.electrical_projections.append(proj_array[synapse_ind])

    return proj_array  


##############################################################################################

def make_target_dict(cell_object,
                     target_segs):
    '''This method constructs the dictionary whose keys are the names of target segment groups or individual segments and the corresponding values are dictionaries
    with keys 'LengthDist' and 'SegList', as returned by the get_seg_lengths. Input arguments are as follows:

    cell_object - object created using libNeuroML API which corresponds to the target cell;
    target_segs - a dictionary in the format returned by the method extract_seg_ids(); the keys are the ids of target segment groups or names of individual segments and 
    the values are lists of corresponding target segment ids.'''

    targetDict = {}
    for target in target_segs.keys():
        targetDict[target] = {}
        lengths, segment_list = get_seg_lengths(cell_object, target_segs[target])
        targetDict[target]['LengthDist'] = lengths
        targetDict[target]['SegList'] = segment_list
    return targetDict


############################################################################################################################

def get_target_cells(population,
                     fraction_to_target,
                     list_of_xvectors=None,
                     list_of_yvectors=None,
                     list_of_zvectors=None):

    '''This method returns the list of target cells according to which fraction of randomly selected cells is targeted and whether these cells are localized in the specific 
    rectangular regions of the network. These regions are specified by list_of_xvectors, list_of_yvectors and list_of_zvectors. These lists must have the same length.

    The input variable list_of_xvectors stores the lists whose elements define the left and right margins of the target rectangular regions along the x dimension.

    Similarly, the input variables list_of_yvectors and list_of_zvectors store the lists whose elements define the left and right margins of the target rectangular regions along
    the y and z dimensions, respectively.'''

    if list_of_xvectors == None or list_of_yvectors == None or list_of_zvectors == None:

        target_cells = random.sample(range(population.size), int(round(fraction_to_target * population.size)))

    else:

        cell_instances = population.instances

        region_specific_targets_per_cell_group = []

        for region in range(0, len(list_of_xvectors)):

            for cell in range(0, len(cell_instances)):

                cell_inst = cell_instances[cell]

                location = cell_inst.location

                if (list_of_xvectors[region][0] < location.x) and \
                    (location.x < list_of_xvectors[region][1]):
                        ########## assumes that the surface of the cortical column starts at the zero level and deeper layers are found at increasingly negative values
                        if (list_of_yvectors[region][0] > location.y) and \
                            (location.y > list_of_yvectors[region][1]):

                            if (list_of_zvectors[region][0] < location.z) and \
                                (location.z < list_of_zvectors[region][1]):

                                region_specific_targets_per_cell_group.append(cell)

        target_cells = random.sample(region_specific_targets_per_cell_group, int(round(fraction_to_target * len(region_specific_targets_per_cell_group))))


    return target_cells


##############################################################################################

def get_seg_lengths(cell_object,
                    target_segments):

    '''This method constructs the cumulative distribution of target segments and the corresponding list of target segment ids.
      Input arguments: cell_object - object created using libNeuroML API which corresponds to the target cell; target_segments - the list of target segment ids. '''

    cumulative_length_dist = []
    segment_list = []
    totalLength = 0
    for seg in cell_object.morphology.segments:
        for target_seg in target_segments:
            if target_seg == seg.id:

                if seg.distal != None:
                    xd = seg.distal.x
                    yd = seg.distal.y
                    zd = seg.distal.z

                if seg.proximal != None:
                    xp = seg.proximal.x
                    yp = seg.proximal.y
                    zp = seg.proximal.z
                else:
                    if seg.parent != None:
                        get_segment_parent = seg.parent
                        get_segment_parent_id = get_segment_parent.segments
                        for segment_parent in cell_object.morphology.segments:
                            if segment_parent.id == get_segment_parent_id:
                                xp = segment_parent.distal.x
                                yp = segment_parent.distal.y
                                zp = segment_parent.distal.z
                dist = [xd, yd, zd]
                prox = [xp, yp, zp] 
                length = math.sqrt(sum([(a - b) ** 2 for a, b in zip(dist, prox)])) 

                segment_list.append(target_seg)
                totalLength = totalLength + length
                cumulative_length_dist.append(totalLength)

    return cumulative_length_dist, segment_list


##############################################################################################

def extract_seg_ids(cell_object,
                    target_compartment_array,
                    targeting_mode):

    '''This method extracts the segment ids that map on the target segment groups or individual segments. 
       cell_object is the loaded cell object using neuroml.loaders.NeuroMLLoader, target_compartment_array is an array of target compartment names (e.g. segment group ids or individual segment names) and targeting_mode is one of the strings: "segments" or "segGroups". '''

    segment_id_array = []
    segment_group_array = {}
    cell_segment_array = []
    for segment in cell_object.morphology.segments:
        segment_id_array.append(segment.id)   
        segment_name_and_id = []
        segment_name_and_id.append(segment.name)
        segment_name_and_id.append(segment.id)
        cell_segment_array.append(segment_name_and_id)
    for segment_group in cell_object.morphology.segment_groups:
        pooled_segment_group_data = {}
        segment_list = []
        segment_group_list = []
        for member in segment_group.members:
            segment_list.append(member.segments)
        for included_segment_group in segment_group.includes:
            segment_group_list.append(included_segment_group.segment_groups)


        pooled_segment_group_data["segments"] = segment_list
        pooled_segment_group_data["groups"] = segment_group_list
        segment_group_array[segment_group.id] = pooled_segment_group_data  


    target_segment_array = {}

    found_target_groups = []

    if targeting_mode == "segments":

        for segment_counter in range(0, len(cell_segment_array)):
            for target_segment in range(0, len(target_compartment_array)):
                if cell_segment_array[segment_counter][0] == target_compartment_array[target_segment]: 
                    target_segment_array[target_compartment_array[target_segment]] = [cell_segment_array[segment_counter][1]]
                    found_target_groups.append(target_compartment_array[target_segment])


    if targeting_mode == "segGroups":

        for segment_group in segment_group_array.keys():
            for target_group in range(0, len(target_compartment_array)):
                if target_compartment_array[target_group] == segment_group:
                    segment_target_array = []
                    found_target_groups.append(target_compartment_array[target_group])
                    if segment_group_array[segment_group]["segments"] != []:
                        for segment in segment_group_array[segment_group]["segments"]:
                            segment_target_array.append(segment)
                    if segment_group_array[segment_group]["groups"] != []:
                        for included_segment_group in segment_group_array[segment_group]["groups"]:
                            for included_segment_group_segment in segment_group_array[included_segment_group]["segments"]:
                                segment_target_array.append(included_segment_group_segment)
                    target_segment_array[target_compartment_array[target_group]] = segment_target_array


    if len(found_target_groups) != len(target_compartment_array):

        groups_not_found = list(set(target_compartment_array)- set(found_target_groups))

        opencortex.print_comment_v("Error in method opencortex.build.extract_seg_ids(): target segments or segment groups in %s are not found in %s. Execution will terminate."
                                   % (groups_not_found, cell_object.id))

        quit()

    return target_segment_array       


######################################################################################

def get_target_segments(seg_specifications,
                        subset_dict):

    '''This method generates the list of target segments and target fractions per cell according to two types of input dictionaries:
    seg_specifications - a dictionary in the format returned by make_target_dict(); keys are target group names or individual segment names
    and the corresponding values are dictionaries with keys 'LengthDist' and 'SegList', as returned by the get_seg_lengths;
    subset_dict - a dictionary whose keys are target group names or individual segment names; each key stores the corresponding number of connections per target group.'''


    target_segs_per_cell = []
    target_fractions_along_per_cell = []

    if isinstance(subset_dict, dict):

        for target_group in subset_dict.keys():

            no_per_target_group = subset_dict[target_group]

            if target_group in seg_specifications.keys():

                target_segs_per_group = []

                target_fractions_along_per_group = []

                cumulative_length_dist = seg_specifications[target_group]['LengthDist']

                segment_list = seg_specifications[target_group]['SegList']

                not_selected = True

                while not_selected:

                    p = random.random()

                    loc = p * cumulative_length_dist[-1]

                    if len(segment_list) == len(cumulative_length_dist):

                        for seg_index in range(0, len(segment_list)):

                            current_dist_value = cumulative_length_dist[seg_index]

                            if seg_index == 0:

                                previous_dist_value = 0

                            else:

                                previous_dist_value = cumulative_length_dist[seg_index-1]

                            if loc > previous_dist_value and loc < current_dist_value:

                                segment_length = current_dist_value-previous_dist_value

                                length_within_seg = loc-previous_dist_value

                                post_fraction_along = float(length_within_seg) / segment_length

                                target_segs_per_group.append(segment_list[seg_index])

                                target_fractions_along_per_group.append(post_fraction_along)

                                break

                    if len(target_segs_per_group) == no_per_target_group:
                        not_selected = False
                        break

                target_segs_per_cell.extend(target_segs_per_group)

                target_fractions_along_per_cell.extend(target_fractions_along_per_group)


    if isinstance(subset_dict, float) or isinstance(subset_dict, int):

        total_num_per_target_groups = int(subset_dict)

        not_selected = True

        while not_selected:

            random_target_group = random.sample(seg_specifications.keys(), 1)[0]

            cumulative_length_dist = seg_specifications[random_target_group]['LengthDist']

            segment_list = seg_specifications[random_target_group]['SegList']

            p = random.random()

            loc = p * cumulative_length_dist[-1]

            if len(segment_list) == len(cumulative_length_dist):

                for seg_index in range(0, len(segment_list)):

                    current_dist_value = cumulative_length_dist[seg_index]

                    if seg_index == 0:

                        previous_dist_value = 0

                    else:

                        previous_dist_value = cumulative_length_dist[seg_index-1]

                    if loc > previous_dist_value and loc < current_dist_value:

                        segment_length = current_dist_value-previous_dist_value

                        length_within_seg = loc-previous_dist_value

                        post_fraction_along = float(length_within_seg) / segment_length

                        target_segs_per_cell.append(segment_list[seg_index])

                        target_fractions_along_per_cell.append(post_fraction_along)

                        break

            if len(target_segs_per_cell) == total_num_per_target_groups:

                not_selected = False

                break

    return target_segs_per_cell, target_fractions_along_per_cell


###########################################################################################################################

def get_pre_and_post_segment_ids(proj):

    '''This method extracts the lists of pre and post segment ids per given projection. Can be used when substituting the cell types from one NeuroML2 network to the other.

    return pre_segment_ids, post_segment_ids'''

    pre_segment_ids = []

    post_segment_ids = []

    if hasattr(proj, 'connection_wds'):

        if proj.connection_wds != []:

            connections = proj.connection_wds

    elif  hasattr(proj, 'connections'):

        if proj.connections != []:

            connections = proj.connections

    elif hasattr(proj, 'electrical_connection_instances'):

        if proj.electrical_connection_instances != []:

            connections = proj.electrical_connection_instances

    else:

        if proj.electrical_connections != []:

            connections = proj.electrical_connections

    for conn_counter in range(0, len(connections)):

        connection = connections[conn_counter]

        pre_segment_ids.append(connection.pre_segment_id)

        post_segment_ids.append(connection.post_segment_id)

    return pre_segment_ids, post_segment_ids

##################################################################################################################################


# Helper method which will be made redundant with a better generated Python API...
def _get_cells_of_all_known_types(nml_doc):

    all_cells_known = []
    all_cells_known.extend(nml_doc.cells)
    all_cells_known.extend(nml_doc.izhikevich_cells)
    all_cells_known.extend(nml_doc.izhikevich2007_cells)
    all_cells_known.extend(nml_doc.iaf_cells)
    all_cells_known.extend(nml_doc.iaf_ref_cells)

    return all_cells_known

# Helper method which will be made redundant with a better generated Python API...
def _get_channels_of_all_known_types(nml_doc):

    all_channels = []
    all_channels.extend(nml_doc.ion_channel)
    all_channels.extend(nml_doc.ion_channel_hhs)
    all_channels.extend(nml_doc.ion_channel_kses)
    all_channels.extend(nml_doc.decaying_pool_concentration_models)
    all_channels.extend(nml_doc.fixed_factor_concentration_models)
    all_channels.extend(nml_doc.ComponentType)

    return all_channels

# Helper method which will be made redundant with a better generated Python API...
def _add_to_neuroml_doc(nml_doc, element):

    if isinstance(element, neuroml.Cell):
        nml_doc.cells.append(element)
    elif isinstance(element, neuroml.IzhikevichCell):
        nml_doc.izhikevich_cells.append(element)
    elif isinstance(element, neuroml.Izhikevich2007Cell):
        nml_doc.izhikevich2007_cells.append(element)
    elif isinstance(element, neuroml.IafRefCell):
        nml_doc.iaf_ref_cells.append(element)
    elif isinstance(element, neuroml.IafCell):
        nml_doc.iaf_cells.append(element)

    elif isinstance(element, neuroml.IonChannelKS):
        nml_doc.ion_channel_kss.append(element)
    elif isinstance(element, neuroml.IonChannelHH):
        nml_doc.ion_channel_hhs.append(element)
    elif isinstance(element, neuroml.IonChannel):
        nml_doc.ion_channel.append(element)
    elif isinstance(element, neuroml.FixedFactorConcentrationModel):
        nml_doc.fixed_factor_concentration_models.append(element)
    elif isinstance(element, neuroml.ComponentType):
        nml_doc.ComponentType.append(element)


# Add to list of files to be copied when the network is saved
def _copy_to_dir_for_model(nml_doc, file_name):

    to_be_copied_on_save.append(file_name)
    
    
# Save included cell/channel files to specific dir when the network is saved
def _finalise_copy_to_dir_for_model(nml_doc, target_dir='./'):
    for file_name in to_be_copied_on_save:

        dir_for_model = target_dir+nml_doc.id
        if not os.path.isdir(dir_for_model):
            os.mkdir(dir_for_model)

        shutil.copy(file_name, dir_for_model)


##########################################################################################   

def copy_nml2_source(dir_to_project_nml2,
                     primary_nml2_dir,
                     electrical_synapse_tags,
                     chemical_synapse_tags,
                     extra_channel_tags=[]):

    '''This method copies the individual NeuroML2 model components from the primary source dir to corresponding component folders in the target dir: "synapses", "gapJunctions",

    "channels" and "cells" are created in the dir_to_project_nml2.'''

    full_path_to_synapses = os.path.join(dir_to_project_nml2, "synapses")

    if not os.path.exists(full_path_to_synapses):

        os.makedirs(full_path_to_synapses)

    full_path_to_gap_junctions = os.path.join(dir_to_project_nml2, "gapJunctions")

    if not os.path.exists(full_path_to_gap_junctions):

        os.makedirs(full_path_to_gap_junctions)

    full_path_to_channels = os.path.join(dir_to_project_nml2, "channels")

    if not os.path.exists(full_path_to_channels):

        os.makedirs(full_path_to_channels)

    full_path_to_cells = os.path.join(dir_to_project_nml2, "cells")

    if not os.path.exists(full_path_to_cells):

        os.makedirs(full_path_to_cells)

    opencortex.print_comment_v("Will be copying cell component files from %s to %s" % (primary_nml2_dir, full_path_to_cells))

    opencortex.print_comment_v("Will be copying channel component files from %s to %s" % (primary_nml2_dir, full_path_to_channels))

    opencortex.print_comment_v("Will be copying synapse component files from %s to %s" % (primary_nml2_dir, full_path_to_synapses))

    opencortex.print_comment_v("Will be copying gap junction component files from %s to %s" % (primary_nml2_dir, full_path_to_gap_junctions))

    src_files = os.listdir(primary_nml2_dir)

    for file_name in src_files:

        full_file_name = os.path.join(primary_nml2_dir, file_name)

        if '.cell.nml' in file_name:

            shutil.copy(full_file_name, full_path_to_cells)

            continue

        for elect_tag in electrical_synapse_tags:

            if elect_tag in file_name:

                shutil.copy(full_file_name, full_path_to_gap_junctions)

                continue

        for chem_tag in chemical_synapse_tags:

            if chem_tag in file_name:

                shutil.copy(full_file_name, full_path_to_synapses)

                continue

        if '.channel.nml' in file_name:

            shutil.copy(full_file_name, full_path_to_channels)

            continue

        if extra_channel_tags != []:  

            for channel_tag in extra_channel_tags:

                if channel_tag in file_name:

                    shutil.copy(full_file_name, full_path_to_channels)



def _include_neuroml2_cell(nml_doc, cell_nml2_path, cell_id, channels_also=True):
    
    '''
    This could be called from opencortex.core
    '''

    nml2_doc_cell = pynml.read_neuroml2_file(cell_nml2_path, include_includes=False)

    for cell in _get_cells_of_all_known_types(nml2_doc_cell):
        if cell.id == cell_id:
            all_cells[cell_id] = cell

            _copy_to_dir_for_model(nml_doc, cell_nml2_path)
            
            new_file = '%s/%s' % (nml_doc.id, os.path.basename(cell_nml2_path))
            
            if not new_file in all_included_files:
                nml_doc.includes.append(neuroml.IncludeType(new_file)) 
                all_included_files.append(new_file)

            if channels_also:
                nml_file_dir = os.path.dirname(cell_nml2_path) if len(os.path.dirname(cell_nml2_path))>0 else '.'
                for included in nml2_doc_cell.includes:

                    incl_file = '%s/%s'%(nml_file_dir,included.href)
                    _copy_to_dir_for_model(nml_doc, incl_file)
                    
                    new_file = '%s/%s' % (nml_doc.id, os.path.basename(included.href))
                    
                    if not new_file in all_included_files:
                        nml_doc.includes.append(neuroml.IncludeType(new_file))
                        all_included_files.append(new_file)
                    
    cell_ids_vs_nml_docs[cell_id] = nml2_doc_cell


#########################################################################################

def _add_cell_and_channels(nml_doc, cell_nml2_rel_path, cell_id, use_prototypes=True):

    if use_prototypes:

        cell_nml2_path = os.path.dirname(__file__) + "/../../NeuroML2/prototypes/" + cell_nml2_rel_path

        opencortex.print_comment_v("Translated %s to %s" % (cell_nml2_rel_path, cell_nml2_path))

    else:

        cell_nml2_path = cell_nml2_rel_path

    nml2_doc_cell = pynml.read_neuroml2_file(cell_nml2_path, include_includes=False)

    for cell in _get_cells_of_all_known_types(nml2_doc_cell):
        if cell.id == cell_id:
            all_cells[cell_id] = cell

            _copy_to_dir_for_model(nml_doc, cell_nml2_path)
            new_file = '%s/%s.cell.nml' % (nml_doc.id, cell_id)
            if not new_file in all_included_files:
                nml_doc.includes.append(neuroml.IncludeType(new_file)) 
                all_included_files.append(new_file)

            for included in nml2_doc_cell.includes:

                if '../channels/' in included.href:

                    path_included = included.href.split("/")

                    channel_file = path_included[-1]

                    old_loc = '../../channels/%s' % channel_file

                elif '..\channels\'' in included.href:

                    path_included = included.href.split("\"")

                    channel_file = path_included[-1]

                    old_loc = "..\..\channels\%s'" % channel_file

                else:

                    channel_file = included.href

                    old_loc = '%s/%s' % (os.path.dirname(os.path.abspath(cell_nml2_path)), channel_file)

                _copy_to_dir_for_model(nml_doc, old_loc)
                new_loc = '%s/%s' % (nml_doc.id, channel_file)
                
                if not new_loc in all_included_files:
                    nml_doc.includes.append(neuroml.IncludeType(new_loc))
                    all_included_files.append(new_loc)


    nml2_doc_cell_full = pynml.read_neuroml2_file(cell_nml2_path, include_includes=True)

    cell_ids_vs_nml_docs[cell_id] = nml2_doc_cell_full


####################################################################################################################################### 

def remove_component_dirs(dir_to_project_nml2,
                          list_of_cell_ids,
                          extra_channel_tags=None):

    '''This method removes the sufolder strings of NeuroML2 component types (if they exist) from the 'includes' of each NeuroML2 cell in the target dir.

    Target directory is specified by the input argument dir_to_project_nml2.'''

    list_of_cell_file_names = []

    for cell_id in list_of_cell_ids:

        list_of_cell_file_names.append(cell_id + ".cell.nml")

    for cell_file_name in list_of_cell_file_names:

        full_path_to_cell = os.path.join(dir_to_project_nml2, cell_file_name)

        nml2_doc_cell = pynml.read_neuroml2_file(full_path_to_cell, include_includes=False)

        for included in nml2_doc_cell.includes:

            if '.channel.nml' in included.href:

                if '../channels/' in included.href:

                    split_href = included.href.split("/")

                    included.href = split_href[-1]

                    continue

                if '..\channels\'' in included.href:

                    split_href = included.href.split("\'")

                    included.href = split_href[-1]

                    continue

            else:

                if extra_channel_tags != None:

                    for channel_tag in included.href:

                        if channel_tag in included.href:

                            if '../channels/' in included.href:

                                split_href = included.href.split("/")

                                included.href = split_href[-1]

                                break

                            if '..\channels\'' in included.href:

                                split_href = included.href.split("\'")

                                included.href = split_href[-1]

                                break

        pynml.write_neuroml2_file(nml2_doc_cell, full_path_to_cell)  


##############################################################################################

def add_synapses(nml_doc, nml2_path, synapse_list, synapse_tag=True):

    for synapse in synapse_list: 

        if synapse_tag:

            _copy_to_dir_for_model(nml_doc, os.path.join(nml2_path, "%s.synapse.nml" % synapse))

            new_file = '%s/%s.synapse.nml' % (nml_doc.id, synapse)

        else:

            _copy_to_dir_for_model(nml_doc, os.path.join(nml2_path, "%s.nml" % synapse))

            new_file = '%s/%s.nml' % (nml_doc.id, synapse)

        nml_doc.includes.append(neuroml.IncludeType(new_file)) 




################################################################################    

def _add_pulse_generator(nml_doc, id, delay, duration, amplitude):

    """
    Adds a <pulseGenerator> element to the document. See the definition of the 
    behaviour of this here: https://www.neuroml.org/NeuroML2CoreTypes/Inputs.html#pulseGenerator
    
    Returns the class created.
    """
    
    pg = neuroml.PulseGenerator(id=id,
                                delay=delay,
                                duration=duration,
                                amplitude=amplitude)

    nml_doc.pulse_generators.append(pg)

    return pg



##############################################################################################

def _add_poisson_firing_synapse(nml_doc, id, average_rate, synapse_id):
    
    """
    Adds a <poissonFiringSynapse> element to the document. See the definition of the 
    behaviour of this here: https://www.neuroml.org/NeuroML2CoreTypes/Inputs.html#poissonFiringSynapse
    
    Returns the class created.
    """

    pfs = neuroml.PoissonFiringSynapse(id=id,
                                       average_rate=average_rate,
                                       synapse=synapse_id, 
                                       spike_target="./%s" % synapse_id)

    nml_doc.poisson_firing_synapses.append(pfs)

    return pfs



#########################################################################

def _add_transient_poisson_firing_synapse(nml_doc, id, average_rate, delay, duration, synapse_id):

    """
    Adds a <transientPoissonFiringSynapse> element to the document. See the definition of the 
    behaviour of this here: https://www.neuroml.org/NeuroML2CoreTypes/Inputs.html#transientPoissonFiringSynapse
    
    Returns the class created.
    """
    
    pfs = neuroml.TransientPoissonFiringSynapse(id=id,
                                                average_rate=average_rate,
                                                delay=delay,
                                                duration=duration,
                                                synapse=synapse_id, 
                                                spike_target="./%s" % synapse_id)

    nml_doc.transient_poisson_firing_synapses.append(pfs)

    return pfs



##############################################################################################

def _add_spike_source_poisson(nml_doc, id, start, duration, rate):

    """
    Adds a <SpikeSourcePoisson> element to the document. See the definition of the 
    behaviour of this here: https://www.neuroml.org/NeuroML2CoreTypes/PyNN.html#SpikeSourcePoisson
    
    Returns the class created.
    """
    
    ssp = neuroml.SpikeSourcePoisson(id=id,
                                     start=start,
                                     duration=duration,
                                     rate=rate)

    nml_doc.SpikeSourcePoisson.append(ssp)

    return ssp

##############################################################################################

def _add_population_in_rectangular_region(net, 
                                         pop_id, 
                                         cell_id, 
                                         size, 
                                         x_min, 
                                         y_min, 
                                         z_min, 
                                         x_size, 
                                         y_size, 
                                         z_size,
                                         cell_bodies_overlap=True,
                                         store_soma=False,
                                         population_dictionary=None,
                                         cell_diameter_dict=None,
                                         color=None):

    """
    See info at opencortex.core.add_population_in_rectangular_region()

    """

    pop = neuroml.Population(id=pop_id, component=cell_id, type="populationList", size=size)

    if color is not None:
        pop.properties.append(neuroml.Property("color", color))



    # If size == 0, don't add to document, but return placeholder object (with attribute size=0 & id)
    if size > 0:
        net.populations.append(pop)

    if store_soma:

        cellPositions = []

    if (not cell_bodies_overlap) and (population_dictionary == None or cell_diameter_dict == None):

        error = "Error! Method opencortex.build.%s() called with cell_bodies_overlap set to False but population_dictionary or cell_diameter_dict is None !" \
                % sys._getframe().f_code.co_name
                
        opencortex.print_comment_v(error)

        raise Exception(error)

    else:

        cellPositions = []

    for i in range(0, size):

        if cell_bodies_overlap:

            inst = neuroml.Instance(id=i)
            pop.instances.append(inst)
            X = x_min + (x_size) * random.random()
            Y = y_min + (y_size) * random.random()
            Z = z_min + (z_size) * random.random()
            inst.location = neuroml.Location(x=str(X), y=str(Y), z=str(Z))

            if store_soma:
                cell_position = []
                cell_position.append(X)  
                cell_position.append(Y)
                cell_position.append(Z)
                cellPositions.append(cell_position)

        else:

            cell_position_found = False

            while not cell_position_found:

                X = x_min + (x_size) * random.random()
                Y = y_min + (y_size) * random.random()
                Z = z_min + (z_size) * random.random()

                try_cell_position = [X, Y, Z]

                count_overlaps = 0

                for cell_index in range(0, len(cellPositions)):

                    cell_loc = cellPositions[cell_index]

                    if distance(try_cell_position, cell_loc) < (cell_diameter_dict[cell_id] + cell_diameter_dict[cell_id]) / 2:

                        count_overlaps += 1

                for pop_id in population_dictionary.keys():

                    if population_dictionary[pop_id] != {}:

                        added_cell_positions = population_dictionary[pop_id]['Positions']

                        cell_component = population_dictionary[pop_id]['PopObj'].component

                        for cell_index in range(0, len(added_cell_positions)):

                            cell_loc = added_cell_positions[cell_index]

                            if distance(try_cell_position, cell_loc) < (cell_diameter_dict[cell_component] + cell_diameter_dict[cell_component]) / 2:

                                count_overlaps += 1

                if count_overlaps == 0:

                    inst = neuroml.Instance(id=i)
                    pop.instances.append(inst)
                    inst.location = neuroml.Location(x=str(X), y=str(Y), z=str(Z))

                    cellPositions.append(try_cell_position)

                    cell_position_found = True  

    if store_soma:

        return pop, cellPositions

    else:

        return pop

##############################################################################################

def add_population_in_cylindrical_region(net, 
                                         pop_id, 
                                         cell_id, 
                                         size, 
                                         cyl_radius,
                                         lower_bound_dim3,
                                         upper_bound_dim3,
                                         base_dim1='x',
                                         base_dim2='z',
                                         cell_bodies_overlap=True,
                                         store_soma=False,
                                         population_dictionary=None,
                                         cell_diameter_dict=None,
                                         num_of_polygon_sides=None,
                                         positions_of_vertices=None,
                                         constants_of_sides=None,
                                         color=None):

    '''Method which create a cell population in the  NeuroML2 network and distributes these cells in the cylindrical region. Input arguments are as follows:

    net - reference to the libNeuroML network object;

    pop_id - population id;

    cell_id - cell component id;

    size - size of a population;

    cyl_radius - radius of a cylindrical column in which cells will be distributed;

    lower_bound_dim3 - lower bound of the cortical column height; 

    upper_bound_dim3 - upper bound of the cortical column height;

    base_dim1 - specifies which of the 'x', 'y' and 'z' axis corresponds to the first dimension of the transverse plane of the cortical column;

    base_dim2 - specifies which of the 'x', 'y' and 'z' axis corresponds to the second dimension of the transverse plane of the cortical column; 

    cell_bodies_overlap -  boolean value which defines whether cell somata can overlap; default is set to True;

    store_soma -boolean value which specifies whether soma positions have to be stored in the output array; default is set to False;

    population_dictionary - optional argument in the format returned by add_populations_in_rectangular_layers; default value is None but it must be specified when cell_bodies_overlap
    is set to False;

    cell_diameter_dict - optional argument in the format {'cell_id1': soma diameter of type 'float', 'cell_id2': soma diameter of type 'float'}; default is None but it must be
    specified when cell_bodies_overlap is set to False.

    num_of_polygon_sides - optional argument which specifies the number of sides of regular polygon which is inscribed in the cylindrical column of a given radius; default value
     is   None, thus cylindrical but not polygonal shape is built.

    positions_of_vertices - optional argument which specifies the list of coordinates [dim1, dim2] of vertices of a regular polygon; must be specified if num_of_polygon_sides is not
    None; automatic generation of this list is wrapped inside the utils method add_populations_in_cylindrical_layers(); 

    constants_of_sides - optional argument which specifies the list of y=ax +b coefficients [a, b] which define the lines between vertices of a regular polygon; 
    if y= b for all values then list element should specify as [None, b]; if x= b for all values of y then list element should specify as [b, None]; 
    Note that constants_of_sides must be specified if num_of_polygon_sides is not None; 
    automatic generation of this list is wrapped inside the utils method add_populations_in_cylindrical_layers(); 

    color - optional color, default is None.

    '''

    pop = neuroml.Population(id=pop_id, component=cell_id, type="populationList", size=size)

    if color is not None:
        pop.properties.append(neuroml.Property("color", color))

    if size > 0:

        net.populations.append(pop)

    if store_soma:

        cellPositions = []

    if (num_of_polygon_sides != None) and (positions_of_vertices == None or constants_of_sides == None):

        opencortex.print_comment_v("Error! Method opencortex.build.%s() called with num_of_polygon_sides set to %d but positions_of_vertices or constants_of_sides "
                                   "is None !. Execution will terminate." % num_of_polygon_sides, sys._getframe().f_code.co_name)

        quit()

    if (not cell_bodies_overlap) and (population_dictionary == None or cell_diameter_dict == None):

        opencortex.print_comment_v("Error! Method opencortex.build.%s() called with cell_bodies_overlap set to False but population_dictionary or cell_diameter_dict is None !" 
                                   "Execution will terminate." % sys._getframe().f_code.co_name)

        quit()

    else:

        cellPositions = []

    all_dims = ['x', 'y', 'z']

    for dim in all_dims:

        if dim != base_dim1 or dim != base_dim2:

            all_dims.remove(dim)

    map_xyz = {base_dim1:'dim1', base_dim2:'dim2', all_dims[0]:'dim3'}

    for i in range(0, size):

        if cell_bodies_overlap:

            dim_dict = find_constrained_cell_position(num_of_polygon_sides, cyl_radius, lower_bound_dim3, upper_bound_dim3, positions_of_vertices, constants_of_sides)

            X = dim_dict[map_xyz['x']]

            Y = dim_dict[map_xyz['y']]

            Z = dim_dict[map_xyz['z']]

            inst = neuroml.Instance(id=i)
            pop.instances.append(inst)

            inst.location = neuroml.Location(x=str(X), y=str(Y), z=str(Z))

            if store_soma:

                cell_position = []
                cell_position.append(X)  
                cell_position.append(Y)
                cell_position.append(Z)
                cellPositions.append(cell_position)

        else:

            cell_position_found = False

            while not cell_position_found:

                dim_dict = find_constrained_cell_position(num_of_polygon_sides, cyl_radius, lower_bound_dim3, upper_bound_dim3, positions_of_vertices, constants_of_sides)

                X = dim_dict[map_xyz['x']]

                Y = dim_dict[map_xyz['y']]

                Z = dim_dict[map_xyz['z']]

                try_cell_position = [X, Y, Z]

                count_overlaps = 0

                for cell_index in range(0, len(cellPositions)):

                    cell_loc = cellPositions[cell_index]

                    if distance(try_cell_position, cell_loc) < (cell_diameter_dict[cell_id] + cell_diameter_dict[cell_id]) / 2:

                        count_overlaps += 1

                for pop_id in population_dictionary.keys():

                    if population_dictionary[pop_id] != {}:

                        added_cell_positions = population_dictionary[pop_id]['Positions']

                        cell_component = population_dictionary[pop_id]['PopObj'].component

                        for cell_index in range(0, len(added_cell_positions)):

                            cell_loc = added_cell_positions[cell_index]

                            if distance(try_cell_position, cell_loc) < (cell_diameter_dict[cell_component] + cell_diameter_dict[cell_component]) / 2:

                                count_overlaps += 1

                if count_overlaps == 0:

                    inst = neuroml.Instance(id=i)
                    pop.instances.append(inst)
                    inst.location = neuroml.Location(x=str(X), y=str(Y), z=str(Z))

                    cellPositions.append(try_cell_position)

                    cell_position_found = True  

    if store_soma:

        return pop, cellPositions

    else:

        return pop


##############################################################################################

def find_constrained_cell_position(num_of_polygon_sides, cyl_radius, lower_bound_dim3, upper_bound_dim3, positions_of_vertices, constants_of_sides):
    '''
    Method to find a constrained position of the cell; used inside the method add_population_in_cylindrical_region(). 
    '''

    if num_of_polygon_sides == None:

        found_cell_inside_cylinder = False

        dim3_val = lower_bound_dim3 + (upper_bound_dim3-lower_bound_dim3) * random.random()

        while not found_cell_inside_cylinder:

            dim_dict = {}

            dim1_min = -cyl_radius

            dim2_min = -cyl_radius

            dim1_val = dim1_min + (2 * cyl_radius) * random.random()

            dim2_val = dim2_min + (2 * cyl_radius) * random.random()

            test_point_inside_cylinder = [dim1_val, dim2_val]

            if distance(test_point_inside_cylinder, [0, 0]) <= cyl_radius:

                dim_dict = {'dim1':dim1_val, 'dim2':dim2_val, 'dim3':dim3_val}

                found_cell_inside_cylinder = True  
    else:

        found_constrained_cell_loc = False

        while not found_constrained_cell_loc:

            found_cell_inside_cylinder = False

            dim3_val = lower_bound_dim3 + (upper_bound_dim3-lower_bound_dim3) * random.random()

            while not found_cell_inside_cylinder:

                dim1_min = -cyl_radius

                dim2_min = -cyl_radius

                dim1_val = dim1_min + (2 * cyl_radius) * random.random()

                dim2_val = dim2_min + (2 * cyl_radius) * random.random()

                test_point_inside_cylinder = [dim1_val, dim2_val]

                if distance(test_point_inside_cylinder, [0, 0]) <= cyl_radius:

                    test_point = test_point_inside_cylinder

                    found_cell_inside_cylinder = True

            count_intersections = 0

            for side_index in range(0, len(constants_of_sides)):

                if abs(positions_of_vertices[side_index][1] - positions_of_vertices[side_index-1][1]) > 0.0000001:

                    if test_point[1] < positions_of_vertices[side_index][1] and test_point[1] > positions_of_vertices[side_index-1][1]:

                        opencortex.print_comment_v("Checking a point inside a regular polygon")

                        if constants_of_sides[side_index][0] != None and constants_of_sides[side_index][1] == None:

                            if dim1_val <= constants_of_sides[side_index][0]:

                                count_intersections += 1

                        if constants_of_sides[side_index][0] != None and constants_of_sides[side_index][1] != None:

                            if dim1_val <= (dim2_val - constants_of_sides[side_index][1]) / constants_of_sides[side_index][0]:

                                count_intersections += 1

                    if test_point[1] < positions_of_vertices[side_index-1][1] and test_point[1] > positions_of_vertices[side_index][1]:

                        opencortex.print_comment_v("Checking a point inside a regular polygon")

                        if constants_of_sides[side_index][0] != None and constants_of_sides[side_index][1] == None:

                            if dim1_val <= constants_of_sides[side_index][0]:

                                count_intersections += 1

                        if constants_of_sides[side_index][0] != None and constants_of_sides[side_index][1] != None:

                            if dim1_val <= (dim2_val - constants_of_sides[side_index][1]) / constants_of_sides[side_index][0]:

                                count_intersections += 1

            if  count_intersections == 1:

                dim_dict = {'dim1':dim1_val, 'dim2':dim2_val, 'dim3':dim3_val}

                opencortex.print_comment_v("Selected a cell locus inside regular polygon.")

                found_constrained_cell_loc = True

    return dim_dict


##############################################################################################

def distance(p, q):

    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p, q)]))


##############################################################################################

def get_soma_diameter(cell_name, cell_type=None, dir_to_cell=None):

    '''Method to obtain a diameter of a cell soma. '''

    loaded_cell_array = {}

    if dir_to_cell == None:

        cell_nml_file = '%s.cell.nml' % cell_name

    else:

        cell_nml_file = os.path.join(dir_to_cell, '%s.cell.nml' % cell_name)

    document_cell = neuroml.loaders.NeuroMLLoader.load(cell_nml_file)

    if cell_type != None:
        if cell_type == "cell2CaPools":
            cell_doc = document_cell.cell2_ca_poolses[0]
    else:
        loaded_cell_array[cell_name] = document_cell.cells[0]

    cell_diameter = 0

    for segment in loaded_cell_array[cell_name].morphology.segments:

        if segment.id == 0:

            proximal_diameter = segment.distal.diameter
            distal_diameter = segment.proximal.diameter

            if proximal_diameter > distal_diameter:

                cell_diameter = proximal_diameter

            else:
                cell_diameter = distal_diameter

            break  

    return cell_diameter



##############################################################################################

def add_advanced_inputs_to_population(net, 
                                      id, 
                                      population, 
                                      input_id_list, 
                                      seg_length_dict,
                                      subset_dict,
                                      universal_target_segment,
                                      universal_fraction_along, 
                                      all_cells=False, 
                                      only_cells=None):

    ''' This method distributes the poisson input synapses on the specific segment groups of target cells. Input arguments to this method:

    net- libNeuroML network object;

    id - unique string that tags the input group created by the method;

    population - libNeuroML population object of a target population;

    input_id_list - this is a list that stores lists of poisson synapse ids or pulse generator ids; 
    if len(input_id_list)== (num of target cells) then each target cell, specified by only_cells or all_cells, has a unique list input components;
    if len(input_id_list != num, then add_advanced_inputs_to_population assumes that all cells share the same list of input components and thus uses input_id_list[0].
    Note that all of the input components (e.g. differing in delays) per given list of input components are mapped on the same membrane point on the target segment of a given cell.

    seg_length_dict - a dictionary whose keys are the ids of target segment groups and the values are the segment length dictionaries in the format returned by make_target_dict(); 

    subset_dict - a dictionary whose keys are the ids of target segment groups and the corresponding dictionary values define the desired number of synaptic connections per target    segment group per each postsynaptic cell;

    universal_target_segment - this should be set to None if subset_dict and seg_length_dict are used; alternatively, universal_target_segment specifies a single target segment on
    all of the target cells for all input components; then seg_length_dict and subset_dict must be set to None.

    universal_fraction_along - this should be set to None if subset_dict and seg_length_dict are used; alternatively, universal_target_fraction specifies a single value of 
    fraction along on all of the target segments for all target cells and all input components; then seg_length_dict and subset_dict must bet set to None;

    all_cells - default value is set to False; if all_cells==True then all cells in a given population will receive the inputs;

    only_cells - optional variable which stores the list of ids of specific target cells; cannot be set together with all_cells. '''

    if all_cells and only_cells is not None:
        opencortex.print_comment_v("Error! Method opencortex.build.%s() called with both arguments all_cells and only_cells set!" % sys._getframe().f_code.co_name)
        exit(-1)

    cell_ids = []

    if all_cells:
        cell_ids = range(population.size)
    if only_cells is not None:
        if only_cells == []:
            return
        cell_ids = only_cells

    input_list_array_final = []

    input_counters_final = []

    for input_cell in range(0, len(input_id_list)):

        input_list_array = []

        input_counters = []

        for input_index in range(0, len(input_id_list[input_cell])):

            input_list = neuroml.InputList(id=id + "_%d_%d" % (input_cell, input_index),
                                           component=input_id_list[input_cell][input_index],
                                           populations=population.id)


            input_list_array.append(input_list)

            input_counters.append(0)

        input_list_array_final.append(input_list_array)

        input_counters_final.append(input_counters)

    cell_counter = 0

    for cell_id in cell_ids:

        if len(input_id_list) == len(cell_ids):

            cell_index = cell_counter

        else:

            cell_index = 0

        if subset_dict != None and seg_length_dict == None and universal_target_segment == None and universal_fraction_along == None:

            if None in subset_dict.keys() and len(subset_dict.keys()) == 1:

                for target_point in range(0, subset_dict[None]):

                    for input_index in range(0, len(input_list_array_final[cell_index])):

                        input = neuroml.Input(id=input_counters_final[cell_index][input_index], 
                                              target="../%s/%i/%s" % (population.id, cell_id, population.component), 
                                              destination="synapses")

                        input_list_array_final[cell_index][input_index].input.append(input)

                        input_counters_final[cell_index][input_index] += 1

        elif seg_length_dict != None and subset_dict != None and universal_target_segment == None and universal_fraction_along == None:

            target_seg_array, target_fractions = get_target_segments(seg_length_dict, subset_dict)

            for target_point in range(0, len(target_seg_array)):

                for input_index in range(0, len(input_list_array_final[cell_index])):

                    input = neuroml.Input(id=input_counters_final[cell_index][input_index], 
                                          target="../%s/%i/%s" % (population.id, cell_id, population.component), 
                                          destination="synapses", segment_id="%d" % target_seg_array[target_point], fraction_along="%f" % target_fractions[target_point])

                    input_list_array_final[cell_index][input_index].input.append(input)

                    input_counters_final[cell_index][input_index] += 1

        else:

            for input_index in range(0, len(input_list_array_final[cell_index])):

                input = neuroml.Input(id=input_counters_final[cell_index][input_index], 
                                      target="../%s/%i/%s" % (population.id, cell_id, population.component), 
                                      destination="synapses", segment_id="%d" % universal_target_segment, fraction_along="%f" % universal_fraction_along)

                input_list_array_final[cell_index][input_index].input.append(input)

                input_counters_final[cell_index][input_index] += 1

        cell_counter += 1

    for input_cell in range(0, len(input_list_array_final)):

        for input_index in range(0, len(input_list_array_final[input_cell])):

            net.input_lists.append(input_list_array_final[input_cell][input_index])


    return input_list_array_final


##############################################################################################

def add_projection_based_inputs(net, 
                                id, 
                                population, 
                                input_id_list, 
                                weight_list,
                                synapse_id,
                                seg_length_dict,
                                subset_dict,
                                universal_target_segment,
                                universal_fraction_along, 
                                all_cells=False, 
                                only_cells=None):


    ''' This method builds input projections between the input components and target population. Input arguments to this method:

    net- libNeuroML network object;

    id - unique string that tags the input group created by the method;

    population - libNeuroML population object of a target population;

    input_id_list - this is a list that stores lists of instance ids of SpikeSourcePoisson component types;

    if len(input_id_list)== (num of target cells) then each target cell, specified by only_cells or all_cells, has a unique list input components;
    if len(input_id_list != num, then add_advanced_inputs_to_population assumes that all cells share the same list of input components and thus uses input_id_list[0].
    Note that all of the input components (e.g. differing in delays) per given list of input components are mapped on the same membrane point on the target segment of a given cell.

    weight_list - lists of connection weights for the input components specified by input_id_list; it must take the same format as input_id_list;

    synapse_id - unique synapse id for all input components specified in input_id_list;

    seg_length_dict - a dictionary whose keys are the ids of target segment groups and the values are the segment length dictionaries in the format returned by make_target_dict(); 

    subset_dict - a dictionary whose keys are the ids of target segment groups and the corresponding dictionary values define the desired number of synaptic connections per target    segment group per each postsynaptic cell;

    universal_target_segment - this should be set to None if subset_dict and seg_length_dict are used; alternatively, universal_target_segment specifies a single target segment on
    all of the target cells for all input components; then seg_length_dict and subset_dict must be set to None.

    universal_fraction_along - this should be set to None if subset_dict and seg_length_dict are used; alternatively, universal_target_fraction specifies a single value of 
    fraction along on all of the target segments for all target cells and all input components; then seg_length_dict and subset_dict must bet set to None;

    all_cells - default value is set to False; if all_cells==True then all cells in a given population will receive the inputs;

    only_cells - optional variable which stores the list of ids of specific target cells; cannot be set together with all_cells. '''

    if all_cells and only_cells is not None:
        opencortex.print_comment_v("Error! Method opencortex.build.%s() called with both arguments all_cells and only_cells set!" % sys._getframe().f_code.co_name)
        exit(-1)

    cell_ids = []

    if all_cells:
        cell_ids = range(population.size)
    if only_cells is not None:
        if only_cells == []:
            return
        cell_ids = only_cells

    spike_source_pops_final = []

    spike_source_projections_final = []

    spike_source_counters_final = []

    for input_cell in range(0, len(input_id_list)):

        spike_source_pops = []

        spike_source_projections = []

        spike_source_counters = []

        for input_index in range(0, len(input_id_list[input_cell])):

            spike_source_pop = neuroml.Population(id="Pop_" + id + "_%d_%d" % (input_cell, input_index), 
                                                  component=input_id_list[input_cell][input_index], 
                                                  size=population.size)

            net.populations.append(spike_source_pop)

            proj = neuroml.Projection(id="Proj_%s_%s" % (spike_source_pop.id, population.id), 
                                      presynaptic_population=spike_source_pop.id, 
                                      postsynaptic_population=population.id, 
                                      synapse=synapse_id)

            spike_source_projections.append(proj)

            spike_source_pops.append(spike_source_pop)

            spike_source_counters.append(0)

        spike_source_pops_final.append(spike_source_pops)

        spike_source_counters_final.append(spike_source_counters)

        spike_source_projections_final.append(spike_source_projections)

    cell_counter = 0

    for cell_id in cell_ids:

        if len(input_id_list) == len(cell_ids):

            cell_index = cell_counter

        else:

            cell_index = 0

        if subset_dict != None and seg_length_dict == None and universal_target_segment == None and universal_fraction_along == None:

            if None in subset_dict.keys() and len(subset_dict.keys()) == 1:

                for target_point in range(0, subset_dict[None]):

                    for input_index in range(0, len(spike_source_pops_final[cell_index])):

                        conn = neuroml.ConnectionWD(id=spike_source_counters_final[cell_index][input_index], \
                                                    pre_cell_id="../%s[%s]" % (spike_source_pops_final[cell_index][input_index].id,
                                                    spike_source_counters_final[cell_index][input_index]), \
                                                    post_cell_id="../%s/%i/%s" % (population.id, cell_id, population.component), \
                                                    weight=weight_list[cell_index][input_index],
                                                    delay="0 ms")

                        spike_source_projections_final[cell_index][input_index].connection_wds.append(conn)

                        spike_source_counters_final[cell_index][input_index] += 1

        elif seg_length_dict != None and subset_dict != None and universal_target_segment == None and universal_fraction_along == None:

            target_seg_array, target_fractions = get_target_segments(seg_length_dict, subset_dict)

            for target_point in range(0, len(target_seg_array)):

                for input_index in range(0, len(spike_source_pops_final[cell_index])):

                    conn = neuroml.ConnectionWD(id=spike_source_counters_final[cell_index][input_index], \
                                                pre_cell_id="../%s[%s]" % (spike_source_pops_final[cell_index][input_index].id,
                                                spike_source_counters_final[cell_index][input_index]), \
                                                post_cell_id="../%s/%i/%s" % (population.id, cell_id, population.component), \
                                                post_segment_id="%d" % target_seg_array[target_point],
                                                post_fraction_along="%f" % target_fractions[target_point],
                                                weight=weight_list[cell_index][input_index],
                                                delay="0 ms") 

                    spike_source_projections_final[cell_index][input_index].connection_wds.append(conn)

                    spike_source_counters_final[cell_index][input_index] += 1

        else:

            for input_index in range(0, len(spike_source_pops_final[cell_index])):

                conn = neuroml.ConnectionWD(id=spike_source_counters_final[cell_index][input_index], \
                                            pre_cell_id="../%s[%s]" % (spike_source_pops_final[cell_index][input_index].id, 
                                            spike_source_counters_final[cell_index][input_index]), \
                                            post_cell_id="../%s/%i/%s" % (population.id, cell_id, population.component), \
                                            post_segment_id="%d" % universal_target_segment,
                                            post_fraction_along="%f" % universal_fraction_along,
                                            weight=weight_list[cell_index][input_index],
                                            delay="0 ms") 

                spike_source_projections_final[cell_index][input_index].connection_wds.append(conn)

                spike_source_counters_final[cell_index][input_index] += 1

        cell_counter += 1

    for input_cell in range(0, len(spike_source_projections_final)):

        for input_index in range(0, len(spike_source_projections_final[input_cell])):

            net.projections.append(spike_source_projections_final[input_cell][input_index])


    return spike_source_pops_final

