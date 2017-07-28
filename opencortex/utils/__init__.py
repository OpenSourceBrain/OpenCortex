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
import opencortex.build as oc_build
import operator
import os
import pyneuroml
from pyneuroml import pynml
import pyneuroml.lems
from pyneuroml.lems.LEMSSimulation import LEMSSimulation
import random
import shutil
import sys


##############################################################################################

def add_populations_in_rectangular_layers(net, boundaryDict, popDict, x_vector, z_vector, storeSoma=True, cellBodiesOverlap=True, cellDiameterArray=None): 

    '''This method distributes the cells in rectangular layers. The input arguments:

   net - libNeuroML network object;

   popDict - a dictionary whose keys are unique cell population ids; each key entry stores a tuple of five elements: population size, Layer tag, cell model id, compartmentalization, color; 

   layer tags (of type string) must make up the keys() of boundaryDict;

   boundaryDict have layer pointers as keys; each entry stores the left and right bound of the layer in the list format , e.g. [L3_min, L3_max]

   x_vector - a vector that stores the left and right bounds of the cortical column along x dimension

   y_vector - a vector that stores the left and right bounds of the cortical column along y dimension

   storeSoma - specifies whether soma positions have to be stored in the output array (default is set to True).

   cellBodiesOverlap - boolean value which defines whether cell somata can overlap; default is set to True;

   cellDiameterArray - optional dictionary of cell model diameters required when cellBodiesOverlap is set to False;

   This method returns the dictionary; each key is a unique cell population id and the corresponding value is a dictionary
   which refers to libNeuroML population object (key 'PopObj') and cell position array ('Positions') which by default is None.'''

    return_pops = {}

    for cell_pop in popDict.keys():

        size, layer, cell_model, compartmentalization, color = popDict[cell_pop]

        if size > 0:

            return_pops[cell_pop] = {}

            xl = x_vector[1]-x_vector[0]

            yl = boundaryDict[layer][1]-boundaryDict[layer][0]

            zl = z_vector[1]-z_vector[0]

            if storeSoma:

                pop, cellPositions = oc_build._add_population_in_rectangular_region(net,
                                                                                   cell_pop,
                                                                                   cell_model,
                                                                                   size,
                                                                                   x_vector[0],
                                                                                   boundaryDict[layer][0],
                                                                                   z_vector[0],
                                                                                   xl,
                                                                                   yl,
                                                                                   zl,
                                                                                   cell_bodies_overlap=cellBodiesOverlap,
                                                                                   store_soma=storeSoma,
                                                                                   population_dictionary=return_pops,
                                                                                   cell_diameter_dict=cellDiameterArray,
                                                                                   color=color)

            else:

                pop = oc_build._add_population_in_rectangular_region(net,
                                                                    cell_pop,
                                                                    cell_model,
                                                                    size,
                                                                    x_vector[0],
                                                                    boundaryDict[layer][0],
                                                                    z_vector[0],
                                                                    xl,
                                                                    yl,
                                                                    zl,
                                                                    cell_bodies_overlap=cellBodiesOverlap,
                                                                    store_soma=storeSoma,
                                                                    population_dictionary=return_pops,
                                                                    cell_diameter_dict=cellDiameterArray)

                cellPositions = None

            return_pops[cell_pop]['PopObj'] = pop
            return_pops[cell_pop]['Positions'] = cellPositions
            return_pops[cell_pop]['Compartments'] = compartmentalization

    opencortex.print_comment_v("This is a final list of cell population ids: %s" % return_pops.keys())

    return return_pops


##############################################################################################

def add_populations_in_cylindrical_layers(net, boundaryDict, popDict, radiusOfCylinder, storeSoma=True, cellBodiesOverlap=True, cellDiameterArray=None, numOfSides=None): 

    '''This method distributes the cells in cylindrical layers. The input arguments:

   net - libNeuroML network object;

   popDict - a dictionary whose keys are unique cell population ids; each key entry stores a tuple of five elements: population size, Layer tag, cell model id, compartmentalization, color;  

   layer tags (of type string) must make up the keys() of boundaryDict;

   boundaryDict have layer pointers as keys; each entry stores the left and right bound of the layer in the list format , e.g. [L3_min, L3_max]

   x_vector - a vector that stores the left and right bounds of the cortical column along x dimension

   y_vector - a vector that stores the left and right bounds of the cortical column along y dimension

   radiusOfCylinder - radius of cylindrical column in which cortical cells will be distributed.

   storeSoma - specifies whether soma positions have to be stored in the output array (default is set to True);

   cellBodiesOverlap - boolean value which defines whether cell somata can overlap; default is set to True;

   cellDiameterArray - optional dictionary of cell model diameters required when cellBodiesOverlap is set to False;

   numOfSides - optional argument which specifies the number of sides of regular polygon which is inscribed in the cylindrical column of a given radius; default value is None,
   thus cylindrical but not polygonal shape is built.

   This method returns the dictionary; each key is a unique cell population id and the corresponding value is a dictionary
   which refers to libNeuroML population object (key 'PopObj') and cell position array ('Positions') which by default is None.'''

    if numOfSides != None:

        if numOfSides >= 3: 

            vertex_array = []

            xy_sides = []

            angle_array = np.linspace(0, 2 * math.pi * (1-(1.0 / numOfSides)), numOfSides)

            opencortex.print_comment_v("Generated the angles of the regular polygon with %d sides: %s" % (numOfSides, angle_array))

            for angle in angle_array:

                vertex = []

                x = radiusOfCylinder * math.cos(angle)

                y = radiusOfCylinder * math.sin(angle)

                vertex.append(x)

                vertex.append(y)

                vertex_array.append(vertex)

            opencortex.print_comment_v("Generated the vertices of the regular polygon with %d sides: %s" % (numOfSides, vertex_array))

            for v_ind in range(0, len(vertex_array)):

                v1 = vertex_array[v_ind]

                v2 = vertex_array[v_ind-1]

                if abs(v1[0] - v2[0]) > 0.00000001 and abs(v1[1] -v2[1]) > 0.00000001:

                    A = np.array([[v1[0], 1], [v2[0], 1]])

                    b = np.array([v1[1], v2[1]])

                    xcyc = np.linalg.solve(A, b)

                    xy_sides.append(list(xcyc))

                else:

                    if abs(v1[0] - v2[0]) <= 0.00000001:

                        xy_sides.append([v1[0], None])

                    if abs(v1[1] -v2[1]) <= 0.00000001:

                        xy_sides.append([None, v1[1]])

        else:

            opencortex.print_comment_v("Error! Method opencortex.build.%s() called with numOfSides set to %d but regular polygon must contain at least 3 vertices." 
                                       "Execution will terminate." % sys._getframe().f_code.co_name, numOfSides)

            quit()

    else:

        vertex_array = None

        xy_sides = None

    return_pops = {}

    for cell_pop in popDict.keys():

        size, layer, cell_model, compartmentalization, color = popDict[cell_pop]

        if size > 0:

            return_pops[cell_pop] = {}

            if storeSoma:

                pop, cellPositions = oc_build.add_population_in_cylindrical_region(net=net,
                                                                                   pop_id=cell_pop,
                                                                                   cell_id=cell_model,
                                                                                   size=size,
                                                                                   cyl_radius=radiusOfCylinder,
                                                                                   lower_bound_dim3=boundaryDict[layer][0],
                                                                                   upper_bound_dim3=boundaryDict[layer][1],
                                                                                   cell_bodies_overlap=cellBodiesOverlap,
                                                                                   store_soma=storeSoma,
                                                                                   population_dictionary=return_pops,
                                                                                   cell_diameter_dict=cellDiameterArray,
                                                                                   num_of_polygon_sides=numOfSides,
                                                                                   positions_of_vertices=vertex_array,
                                                                                   constants_of_sides=xy_sides,
                                                                                   color = color)

            else:

                pop = oc_build.add_population_in_cylindrical_region(net=net,
                                                                    pop_id=cell_pop,
                                                                    cell_id=cell_model,
                                                                    size=size,
                                                                    cyl_radius=radiusOfCylinder,
                                                                    lower_bound_dim3=boundaryDict[layer][0],
                                                                    upper_bound_dim3=boundaryDict[layer][1],
                                                                    cell_bodies_overlap=cellBodiesOverlap,
                                                                    store_soma=storeSoma,
                                                                    population_dictionary=return_pops,
                                                                    cell_diameter_dict=cellDiameterArray,
                                                                    num_of_polygon_sides=numOfSides,
                                                                    positions_of_vertices=vertex_array,
                                                                    constants_of_sides=xy_sides)


                cellPositions = None                                                                 

            return_pops[cell_pop]['PopObj'] = pop
            return_pops[cell_pop]['Positions'] = cellPositions
            return_pops[cell_pop]['Compartments'] = compartmentalization

    opencortex.print_comment_v("This is a final list of cell population ids: %s" % return_pops.keys())

    return return_pops


##############################################################################################

def build_projection(net, 
                     proj_counter,
                     proj_type,
                     presynaptic_population, 
                     postsynaptic_population, 
                     synapse_list, 
                     targeting_mode,
                     pre_seg_length_dict,
                     post_seg_length_dict,
                     num_of_conn_dict,
                     distance_dependent_rule=None,
                     pre_cell_positions=None,
                     post_cell_positions=None,
                     delays_dict=None,
                     weights_dict=None):


    '''This method calls the appropriate methods that construct chemical or electrical projections. The input arguments are as follows:

    net - the network object created using libNeuroML ( neuroml.Network() );

    proj_counter - stores the number of projections at any given moment;

    proj_type - 'Chem' or 'Elect' depending on whether the projection is chemical or electrical.

    presynaptic_population - object corresponding to the presynaptic population in the network;

    postsynaptic_population - object corresponding to the postsynaptic population in the network;

    synapse_list - the list of synapse ids that correspond to the individual receptor components on the physical synapse, e.g. the first element is
    the id of the AMPA synapse and the second element is the id of the NMDA synapse; these synapse components will be mapped onto the same location of the target segment;

    targeting_mode - specifies the type of projection: divergent or convergent;

    pre_seg_length_dict - a dictionary whose keys are the ids of presynaptic segment groups and the values are dictionaries in the format returned by make_target_dict();

    post_seg_length_dict - a dictionary whose keys are the ids of target segment groups and the values are dictionaries in the format returned by make_target_dict();

    num_of_conn_dict - a dictionary whose keys are the ids of target segment groups with the corresponding values of type 'int' specifying the number of connections per 
    tarrget segment group per each cell.

    distance_dependent_rule - optional string which defines the distance dependent rule of connectivity - soma to soma distance must be represented by the string character 'r';

    pre_cell_positions- optional array specifying the cell positions for the presynaptic population; the format is an array of [ x coordinate, y coordinate, z coordinate];

    post_cell_positions- optional array specifying the cell positions for the postsynaptic population; the format is an array of [ x coordinate, y coordinate, z coordinate];

    delays_dict - optional dictionary that specifies the delays (in ms) for individual synapse components, e.g. {'NMDA':5.0} or {'AMPA':3.0,'NMDA':5};

    weights_dict - optional dictionary that specifies the weights (in ms) for individual synapse components, e.g. {'NMDA':1} or {'NMDA':1,'AMPA':2}.'''  


    if presynaptic_population.size == 0 or postsynaptic_population.size == 0:
        return None

    proj_array = []
    syn_counter = 0

    synapse_list = list(set(synapse_list))

    for synapse_id in synapse_list:

        if proj_type == 'Elect':

            proj = neuroml.ElectricalProjection(id="Proj%dsyn%d_%s_%s" % (proj_counter, syn_counter, presynaptic_population.id, postsynaptic_population.id),
                                                presynaptic_population=presynaptic_population.id,
                                                postsynaptic_population=postsynaptic_population.id)

            syn_counter += 1
            proj_array.append(proj)

        if proj_type == 'Chem':

            proj = neuroml.Projection(id="Proj%dsyn%d_%s_%s" % (proj_counter, syn_counter, presynaptic_population.id, postsynaptic_population.id), 
                                      presynaptic_population=presynaptic_population.id, 
                                      postsynaptic_population=postsynaptic_population.id, 
                                      synapse=synapse_id)

            syn_counter += 1              
            proj_array.append(proj)


    if distance_dependent_rule == None:

        if proj_type == 'Chem':
            proj_array = oc_build.add_targeted_projection_by_dicts(net,
                                                      proj_array,
                                                      presynaptic_population,
                                                      postsynaptic_population,
                                                      targeting_mode,
                                                      synapse_list,
                                                      pre_seg_length_dict,
                                                      post_seg_length_dict,
                                                      num_of_conn_dict,
                                                      delays_dict,
                                                      weights_dict) 

        if proj_type == 'Elect':
            proj_array = oc_build._add_elect_projection(net,
                                                       proj_array,
                                                       presynaptic_population,
                                                       postsynaptic_population,
                                                       targeting_mode,
                                                       synapse_list,
                                                       pre_seg_length_dict,
                                                       post_seg_length_dict,
                                                       num_of_conn_dict)
    else:

        if proj_type == 'Chem':
            proj_array = oc_build.add_chem_spatial_projection(net,
                                                              proj_array,
                                                              presynaptic_population,
                                                              postsynaptic_population,
                                                              targeting_mode,
                                                              synapse_list,
                                                              pre_seg_length_dict,
                                                              post_seg_length_dict,
                                                              num_of_conn_dict,
                                                              distance_dependent_rule,
                                                              pre_cell_positions,
                                                              post_cell_positions,
                                                              delays_dict,
                                                              weights_dict)

        if proj_type == 'Elect':
            proj_array = oc_build.add_elect_spatial_projection(net,
                                                               proj_array,
                                                               presynaptic_population,
                                                               postsynaptic_population,
                                                               targeting_mode,
                                                               synapse_list,
                                                               pre_seg_length_dict,
                                                               post_seg_length_dict,
                                                               num_of_conn_dict,
                                                               distance_dependent_rule,
                                                               pre_cell_positions,
                                                               post_cell_positions)




    return proj_array, proj_counter



##############################################################################################

def build_connectivity(net,
                       pop_objects,
                       path_to_cells,
                       full_path_to_conn_summary,
                       pre_segment_group_info=[],
                       return_cached_dicts=True,
                       synaptic_scaling_params=None,
                       synaptic_delay_params=None,
                       distance_dependence_params=None,
                       ignore_synapses=[]):

    '''This method calls the appropriate build and utils methods to build connectivity of the NeuroML2 cortical network. Input arguments are as follows:

    net- NeuroML2 network object;

    pop_objects - dictionary of population parameters in the format returned by the utils method add_populations_in_rectangular_layers() or add_populations_in_cylindrical_layers();

    path_to_cells - dir path to the folder where target NeuroML2 .cell.nml files are found;

    full_path_to_conn_sumary - full path to the file which stores the connectivity summary, e.g. file named netConnList in the current working dir, 
    then this string must be "netConnList";

    pre_segment_group_info - input argument of type 'list' which specifies presynaptic segment groups; made to supplement connectivity summary of type netConnList 
    in the Thalamocortical project; default value is []; alternatively it might have one value of type'dict' or several values of type 'dict'; in the former case, 
    dict should contain the fields 'PreSegGroup' and 'ProjType';  in the latter case each dictionary should contain the fields 'PrePop', 'PostPop', 'PreSegGroup'
    and 'ProjType', which uniquely specifies one presynaptic segment group per pair of cell populations per type of projection.

    return_cached_dicts -  boolean-type argument which specifies whether build_connectivity returns the cached dictionary of cummulative distributions of segment lengths 
    for all of the target segment groups. If return_cached_dicts is set to True the last output argument that is returned by build_connectivity is a cached target dictionary;
    the cached target dictionary is specifically built by the method check_cached_dicts inside utils;

    synaptic_scaling_params - optional input argument, default value is None. Alternatively, it takes the format of

    [{'weight':2.0,'synComp':'all'}]                     or

    [{'weight':2.5,'synComp':'GABAA','synEndsWith':[],'targetCellGroup':[]},
     {'weight':0.5,'synComp':'Syn_Elect_DeepPyr_DeepPyr','synEndsWith':[],'targetCellGroup':['CG3D_L5']} ]. Tailored for the NeuroML2 Thalamocortical project.

    synaptic_delay_params - optional input argument, default value is None. Alternatively ,it takes the format similar to the synaptic_scaling_params:

    [{'delay':0.05,'synComp':'all'}]                     or

    [{'delay':0.10,'synComp':'GABAA','synEndsWith':[],'targetCellGroup':[]},
     {'delay':0.05,'synComp':'Syn_Elect_DeepPyr_DeepPyr','synEndsWith':[],'targetCellGroup':['CG3D_L5']} ]. Tailored for the NeuroML2 Thalamocortical project.

    distance_dependent_params - optional input argument, default value is None. Alternatively, it take the format of

    [{'PrePopID':'Pop1','PostPopID':'Pop2','DistDependConn':'- 17.45 + 18.36 / (math.exp((r-267.)/39.) +1)','Type':'Elect'}]. '''

    final_synapse_list = []

    final_proj_array = []

    cached_target_dict = {} 

    proj_counter = 0

    for prePop in pop_objects.keys():

        preCellObject = pop_objects[prePop]

        for postPop in pop_objects.keys():

            postCellObject = pop_objects[postPop]

            if preCellObject['PopObj'].size != 0 and postCellObject['PopObj'].size != 0:

                proj_summary = read_connectivity(prePop,
                                                 postPop,
                                                 full_path_to_conn_summary,
                                                 ignore_synapses=ignore_synapses)

                if proj_summary != []:

                    for proj_ind in range(0, len(proj_summary)):

                        projInfo = proj_summary[proj_ind]

                        target_comp_groups = projInfo['LocOnPostCell']

                        synapseList = projInfo['SynapseList']   

                        final_synapse_list.extend(projInfo['SynapseList'])

                        if 'NumPerPostCell' in projInfo:

                            targetingMode = 'convergent'

                            mode_string = 'NumPerPostCell'

                        if 'NumPerPreCell' in projInfo:

                            targetingMode = 'divergent'

                            mode_string = 'NumPerPreCell'

                        if synaptic_scaling_params != None:

                            weights = parse_weights(synaptic_scaling_params, postPop, synapseList)

                        else:

                            weights = None

                        if synaptic_delay_params != None:

                            delays = parse_delays(synaptic_delay_params, postPop, synapseList)

                        else:

                            delays = None

                        if distance_dependence_params != None:

                            dist_par = parse_distance_dependent_rule(distance_dependence_params, prePop, postPop, projInfo['Type'])

                        else:

                            dist_par = None

                        ### assumes one target segment group per given projection in the format of netConnLists   ### 
                        subset_dict = {}

                        subset_dict[target_comp_groups] = float(projInfo[mode_string])

                        target_comp_groups = [target_comp_groups]
                        #############################################################################################

                        PostSegLengthDict, cached_target_dict = check_cached_dicts(postCellObject['PopObj'].component,
                                                                                   cached_target_dict,
                                                                                   target_comp_groups,
                                                                                   path_to_nml2=path_to_cells)  
                        if pre_segment_group_info != []:

                            passed_pre_seg_groups = check_pre_segment_groups(pre_segment_group_info)

                            if not passed_pre_seg_groups:

                                opencortex.print_comment_v("Error: the list pre_segment_group_info was not correctly specified. Execution will terminate.")

                                quit()

                            else:
                                ############# this block is tailored for handling the connectivity summary in the format of netConnList in the Thalamocortical project.
                                if len(pre_segment_group_info) == 1:

                                    if pre_segment_group_info[0]['ProjType'] == projInfo['Type']:

                                        PreSegLengthDict, cached_target_dict = check_cached_dicts(preCellObject['PopObj'].component,
                                                                                                  cached_target_dict,
                                                                                                  [pre_segment_group_info[0]['PreSegGroup']],
                                                                                                  path_to_nml2=path_to_cells)    
                                    else:

                                        if projInfo['Type'] == 'Elect':

                                            PreSegLengthDict, cached_target_dict = check_cached_dicts(preCellObject['PopObj'].component,
                                                                                                      cached_target_dict,
                                                                                                      target_comp_groups,
                                                                                                      path_to_nml2=path_to_cells)  

                                        else:

                                            PreSegLengthDict = None
                                ##############################################################                                                          
                                if len(pre_segment_group_info) > 1:

                                    found_pre_segment_group = False

                                    for proj in range(0, len(pre_segment_group_info)):

                                        check_pre_pop = pre_segment_group_info[proj]['PrePop'] == prePop

                                        check_post_pop = pre_segment_group_info[proj]['PostPop'] == postPop

                                        check_proj_type = pre_segment_group_info[proj]['ProjType'] == projInfo['Type']

                                        if check_pre_pop and check_post_pop and check_proj_type:

                                            PreSegLengthDict, cached_target_dict = check_cached_dicts(preCellObject['PopObj'].component,
                                                                                                      cached_target_dict,
                                                                                                      [pre_segment_group_info[proj]['PreSegGroup']],
                                                                                                      path_to_nml2=path_to_cells) 

                                            found_pre_segment_group = True

                                            break


                                    if not found_pre_segment_group:

                                        PreSegLengthDict = None                                                  
                        else:

                            PreSegLengthDict = None                                                              

                        compound_proj = build_projection(net=net, 
                                                         proj_counter=proj_counter,
                                                         proj_type=projInfo['Type'],
                                                         presynaptic_population=preCellObject['PopObj'], 
                                                         postsynaptic_population=postCellObject['PopObj'], 
                                                         synapse_list=synapseList, 
                                                         targeting_mode=targetingMode,
                                                         pre_seg_length_dict=PreSegLengthDict,
                                                         post_seg_length_dict=PostSegLengthDict,
                                                         num_of_conn_dict=subset_dict,
                                                         distance_dependent_rule=dist_par,
                                                         pre_cell_positions=preCellObject['Positions'],
                                                         post_cell_positions=postCellObject['Positions'],
                                                         delays_dict=delays,
                                                         weights_dict=weights)


                        proj_counter += 1                      
                        final_proj_array.extend(compound_proj)


    final_synapse_list = np.unique(final_synapse_list)

    final_synapse_list = list(final_synapse_list)

    if return_cached_dicts:

        return final_synapse_list, final_proj_array, cached_target_dict

    else:

        return final_synapse_list, final_proj_array


##############################################################################################

def build_probability_based_connectivity(net,
                                         pop_params,
                                         probability_matrix, 
                                         synapse_matrix,
                                         weight_matrix, 
                                         delay_matrix,
                                         tags_on_populations, 
                                         std_weight_matrix=None,
                                         std_delay_matrix=None):

    ''''Method which build network projections based on the probability matrix which specifies the probabilities between given populations. Input arguments:

    net- NeuroML2 network object;

    pop_params - dictionary of population parameters in the format returned by the utils method add_populations_in_rectangular_layers() or
    add_populations_in_cylindrical_layers();

    probability_matrix -  n by n array (list of lists or numpy array) which specifies the connection probabilities between the presynaptic and postsynaptic populations; 
    the first index is for the target population (postsynaptic) and the second index is for the source population (presynaptic); can be 1 by 1 matrix , then probability 
    is applied to  all pairs of populations; probability values must be of type 'float'; 

    synapse_matrix - n by n array (list of lists) which specifies the synapse components per projections; each element has be of the type 'list' because generically
    physical projection can contain multiple synaptic components;

    weight_matrix - n by n array (list of lists or numpy array) which specifies the weight values for projections; each matrix element should be of type 'float' or if synaptic
    components per given projection differ in weights, then a corresponding matrix element must be a list containing 'float' values;

    delay_matrix - n by n array (list of lists or numpy array) which specifies the delay values for projections; each matrix element should be of type 'float' or if synaptic
    components per given projection differ in delays, then a corresponding matrix elment must be a list containing 'float' values;

    tags_on_populations - a list of n strings which tags the network populations; cannot have length larger than the number of populations; index of the population tag in 
    tags_on_populations must correspond to the position of the matrix element in a standard way, e.g. 

    tags_on_populations= [ 'pop1', 'pop2' ], thus 'pop1' index is 0 and 'pop2' index is 1;

                                                    source population (presynaptic)

    target population (postsynaptic)               'pop1'                              'pop2'


    'pop1'                                  (0,0) value in matrix             (0,1) value in matrix


    'pop2'                                  (1,0) value in matrix              (1,1) value in matrix              . This applies to all matrices.


    std_weight_matrix - optional matrix in the format weight_synapse which specifies the corresponding standard deviations of synaptic weights; default is set to None;

    std_delay_matrix - optional matrix in the format delay_synapse which specifies the corresponding standard deviations of synaptic delays; default is set to None.'''

    errors_found = 0

    matrices_with_errors = []

    passed_probs = check_matrix_size_and_type(matrix_of_params=probability_matrix, num_of_pop_tags=len(tags_on_populations), type_of_matrix=float)

    if not passed_probs:

        errors_found += 1

        matrices_with_errors.append('matrix of connection probabilities')

    passed_syns = check_matrix_size_and_type(matrix_of_params=synapse_matrix, num_of_pop_tags=len(tags_on_populations), type_of_matrix=str)

    if not passed_syns:

        errors_found += 1

        matrices_with_errors.append('matrix of synapse component ids')

    passed_weights = check_matrix_size_and_type(matrix_of_params=weight_matrix, num_of_pop_tags=len(tags_on_populations), type_of_matrix=float)

    if not passed_weights:

        errors_found += 1

        matrices_with_errors.append('matrix of mean values for synaptic weights')

    passed_delays = check_matrix_size_and_type(matrix_of_params=delay_matrix, num_of_pop_tags=len(tags_on_populations), type_of_matrix=float)

    if not passed_delays:

        errors_found += 1

        matrices_with_errors.append('matrix of mean values for synaptic delays')

    if std_weight_matrix != None:

        passed_std_weights = check_matrix_size_and_type(matrix_of_params=std_weight_matrix, num_of_pop_tags=len(tags_on_populations), type_of_matrix=float)

        if not passed_std_weights:

            errors_found += 1

            matrices_with_errors.append('matrix of standard deviations for synaptic weights')

    if std_delay_matrix != None:

        passed_std_delays = check_matrix_size_and_type(matrix_of_params=std_delay_matrix, num_of_pop_tags=len(tags_on_populations), type_of_matrix=float)

        if not passed_std_delays:

            errors_found += 1

            matrices_with_errors.append('matrix of standard deviations for synaptic delays')

    if errors_found == 0:

        opencortex.print_comment_v("All of the connectivity matrices were specified correctly.")

    else:

        opencortex.print_comment_v("The connectivity matrices in %s were not specified correctly; execution will terminate." % matrices_with_errors)

        quit()

    proj_array = []

    for pre_pop_id in pop_params.keys():

        for post_pop_id in pop_params.keys():

            found_pre_pop = False

            found_post_pop = False

            for tag_ind in range(0, len(tags_on_populations)):

                tag = tags_on_populations[tag_ind]

                if tag in pre_pop_id:

                    column_index = tag_ind

                    found_pre_pop = True

                if tag in post_pop_id:

                    row_index = tag_ind

                    found_post_pop = True

            if found_pre_pop and found_post_pop:

                pre_pop_obj = pop_params[pre_pop_id]['PopObj']

                post_pop_obj = pop_params[post_pop_id]['PopObj']

                ####################################################################
                prob_val = parse_parameter_value(parameter_matrix=probability_matrix, 
                                                 row_ind=row_index, 
                                                 col_ind=column_index,
                                                 checks_passed=True)

                synapse_list = parse_parameter_value(parameter_matrix=synapse_matrix, 
                                                     row_ind=row_index, 
                                                     col_ind=column_index,
                                                     checks_passed=True) 

                #######################################################################                                  
                if prob_val != 0 and pre_pop_obj.size != 0 and post_pop_obj.size != 0:

                    if synapse_list != None:

                        if not isinstance(synapse_list, list):

                            synapse_list = [synapse_list]
                        ##########################################################################  
                        weight_list = parse_parameter_value(parameter_matrix=weight_matrix, 
                                                            row_ind=row_index, 
                                                            col_ind=column_index,
                                                            checks_passed=True)

                        ###########################################################################  
                        delay_list = parse_parameter_value(parameter_matrix=delay_matrix, 
                                                           row_ind=row_index, 
                                                           col_ind=column_index,
                                                           checks_passed=True)

                        ############################################################################  
                        if std_weight_matrix != None:

                            std_weight_list = parse_parameter_value(parameter_matrix=std_weight_matrix, 
                                                                    row_ind=row_index, 
                                                                    col_ind=column_index,
                                                                    checks_passed=True)

                        else:

                            std_weight_list = None
                        #############################################################################   
                        if std_delay_matrix != None:

                            std_delay_list = parse_parameter_value(parameter_matrix=std_delay_matrix, 
                                                                   row_ind=row_index, 
                                                                   col_ind=column_index,
                                                                   checks_passed=True)

                        else:

                            std_delay_list = None
                        ###############################################################################

                        returned_projs = oc_build.add_probabilistic_projection_list(net=net,
                                                                                    presynaptic_population=pre_pop_obj, 
                                                                                    postsynaptic_population=post_pop_obj, 
                                                                                    synapse_list=synapse_list, 
                                                                                    connection_probability=prob_val,
                                                                                    delay=delay_list,
                                                                                    weight=weight_list,
                                                                                    std_delay=std_delay_list,
                                                                                    std_weight=std_weight_list)

                        if returned_projs != None:

                            opencortex.print_comment_v("Addded a projection between %s and %s" % (pre_pop_id, post_pop_id))

                            proj_array.extend(returned_projs)

                    else:

                        opencortex.print_comment_v("Error in opencortex.utils.build_probability_based_connectivity(): connection probability is not equal to 0 but synapse list is None"
                                                   " for the projection between presynaptic population id=%s and postsynaptic population id= %s. Execution will terminate." % (pre_pop_id, post_pop_id))

                        quit()

            else:

                if not found_pre_pop:

                    opencortex.print_comment_v("Error in build_probability_based_connectivity(): input argument tags_on_populations in build_probability_based_connectivity()"
                                               " does not contain a string tag for the population %s. Execution will terminate." % pre_pop_id) 

                    quit()

                if not found_post_pop:

                    opencortex.print_comment_v("Error in build_probability_based_connectivity(): input argument tags_on_populations in build_probability_based_connectivity()" 
                                               " does not contain a string tag for the population %s. Execution will terminate." % post_pop_id) 

                    quit()

    return proj_array     


##############################################################################################

def read_connectivity(pre_pop,
                      post_pop,
                      path_to_txt_file,
                      ignore_synapses=[]):

    '''Method that reads the txt file in the format of netConnList found in: Thalamocortical/neuroConstruct/pythonScripts/netbuild.'''                   

    proj_summary = []

    with open(path_to_txt_file, 'r') as file:

        lines = file.readlines()

    num_per_post_cell = False

    num_per_pre_cell = False

    for line in lines:

        if 'NumPerPostCell' in line:

            num_per_post_cell = True
            break

        if 'NumPerPreCell' in line:

            num_per_pre_cell = True
            break

    for line in lines:

        split_line = line.split(" ")

        extract_info = []

        for string_index in range(0, len(split_line)):

            if split_line[string_index] != '':

                extract_info.append(split_line[string_index])

        counter = 0    

        if pre_pop in extract_info[0] and post_pop in extract_info[1]:

            proj_info = {}

            proj_info['PreCellGroup'] = pre_pop

            proj_info['PostCellGroup'] = post_pop

            synapse_list = []

            if ',' in extract_info[2]:
                synapse_list_string = extract_info[2].split(',')
            else:
                synapse_list_string = [extract_info[2]]

            for synapse_string in synapse_list_string:

                if '[' and ']' in synapse_string:

                    left = synapse_string.find('[')

                    right = synapse_string.find(']')

                    synapse = synapse_string[left + 1:right]

                    synapse_list.append(synapse)

                    continue

                if '[' in synapse_string:

                    left = synapse_string.find('[')

                    synapse = synapse_string[left + 1:]

                    synapse_list.append(synapse)

                    continue

                if ']' in synapse_string:

                    right = synapse_string.find(']')

                    synapse = synapse_string[0:right]

                    synapse_list.append(synapse)

                    continue  

            for syn in ignore_synapses:
                if syn in synapse_list: synapse_list.remove(syn)


            proj_info['SynapseList'] = synapse_list

            if 'Elect' in extract_info[2]:
                proj_info['Type'] = 'Elect'       
            else:
                proj_info['Type'] = 'Chem'

            if num_per_post_cell:

                proj_info['NumPerPostCell'] = extract_info[3]

            if num_per_pre_cell:

                proj_info['NumPerPreCell'] = extract_info[3]

            if '\n' in extract_info[4]:

                proj_info['LocOnPostCell'] = extract_info[4][0:-1]

            else:

                proj_info['LocOnPostCell'] = extract_info[4]

            proj_summary.append(proj_info)   


    return proj_summary


##############################################################################################

def parse_parameter_value(parameter_matrix, row_ind, col_ind, checks_passed=False):

    '''Method to parse one of the connectivity parameters; by default assumes that checks are carried out by the method check_matrix_size_and_type().'''

    if not checks_passed:

        return None

    else:

        if len(parameter_matrix) == 1:

            parameter_val = parameter_matrix[0]

        else: 

            parameter_val = parameter_matrix[row_ind][col_ind]

        if type(parameter_matrix).__module__ == np.__name__:

            parameter_val = parameter_matrix[row_ind][col_ind]

        return parameter_val


##############################################################################################

def check_matrix_size_and_type(matrix_of_params, num_of_pop_tags, type_of_matrix):

    '''Method to check whether the size and the type of the connection parameter matrix, corresponds to the number of tags provided for the list of populations.'''

    passed = True

    if isinstance(matrix_of_params, list):

        if len(matrix_of_params) == 1:

            if matrix_of_params[0] != None:

                if (not isinstance(matrix_of_params[0], list)):

                    if not isinstance(matrix_of_params[0], type_of_matrix):

                        opencortex.print_comment_v("Error in check_matrix_size(): argument matrix_of_params is a list which contains one element but it is not of type %s;"
                                                   " the current type is %s." % (type_of_matrix, type(matrix_of_params[0])))

                        passed = False

                else:

                    for component_index in range(0, len(matrix_of_params[0])):

                        if not isinstance(matrix_of_params[0][component_index], type_of_matrix):

                            opencortex.print_comment_v("Error in check_matrix_size(): argument matrix_of_params is a list which contains a list but not all of its elements are of type %s;"
                                                       " the current type is %s." % (type_of_matrix, type(matrix_of_params[0][component])))

                            passed = False

        else:

            num_of_elements = 0

            for row_ind in range(0, len(matrix_of_params)):

                for col_ind in range(0, len(matrix_of_params[row_ind])):

                    if (matrix_of_params[row_ind][col_ind] != None):

                        if (not isinstance(matrix_of_params[row_ind][col_ind], list)):

                            if not isinstance(matrix_of_params[row_ind][col_ind], type_of_matrix):

                                opencortex.print_comment_v("Error in check_matrix_size(): argument matrix_of_params is a list of lists (matrix) but there is an element inside which is not"
                                                           " of type %s; the current type is %s." % (type_of_matrix, type(matrix_of_params[row_ind][col_ind])))

                                passed = False

                        else:

                            for component_index in range(0, len(matrix_of_params[row_ind][col_ind])):

                                if not isinstance(matrix_of_params[row_ind][col_ind][component_index], type_of_matrix):

                                    opencortex.print_comment_v("Error in check_matrix_size(): argument matrix_of_params is a list of lists (matrix) but list elements inside are not" 
                                                               "of type %s; the current type is %s." % (type_of_matrix, type(matrix_of_params[row_ind][col_ind][component_index])))

                                    passed = False

                num_of_elements = num_of_elements + len(matrix_of_params[row_ind])

                if len(matrix_of_params[row_ind]) != num_of_pop_tags:

                    opencortex.print_comment_v("Error in check_matrix_size(): each list element of the argument matrix_of_params must have length = %d;"
                                               " the current length is %d." % (num_of_pop_tags, len(matrix_of_params[row_ind])))

                    passed = False

            if num_of_elements != num_of_pop_tags * num_of_pop_tags:

                opencortex.print_comment_v("Error in check_matrix_size(): argument matrix_of_params is a list but not in the format of an %d by %d matrix."
                                           % (num_of_pop_tags, num_of_pop_tags))

                passed = False

    elif type(matrix_of_params).__module__ == np.__name__:

        rows_cols = matrix_of_params.size

        if (rows_cols[0] != num_of_pop_tags or rows_cols[1] != num_of_pop_tags):

            opencortex.print_comment_v("Error in check_matrix_size(): argument matrix_of_params is a numpy array but not of size (%d, %d) (not a square matrix)." % num_of_pop_tags)

            passed = False

    else:

        opencortex.print_comment_v("Error in check_matrix_size(): argument matrix_of_params must be a list or numpy array; the current type is %s."
                                   % type(matrix_of_params))

        passed = False

    return passed


##############################################################################################

def check_cached_dicts(cell_component, cached_dicts, list_of_target_seg_groups, path_to_nml2=None):

    '''This method checks whether information is missing on target segment groups and updates the output dictionary with new information for the target cell component. '''

    segLengthDict = {}

    if cell_component in cached_dicts.keys():

        target_groups_to_include = []

        new_segment_groups = False

        segLengthDict = {}

        for target_group in list_of_target_seg_groups:

            if target_group not in cached_dicts[cell_component]['TargetDict'].keys():

                target_groups_to_include.append(target_group)

                new_segment_groups = True

            else:

                segLengthDict[target_group] = cached_dicts[cell_component]['TargetDict'][target_group]

            if new_segment_groups:

                cellObject = cached_dicts[cell_component]['CellObject']

                target_segments = oc_build.extract_seg_ids(cell_object=cellObject, target_compartment_array=target_groups_to_include, targeting_mode='segGroups') 

                new_seg_length_dict = oc_build.make_target_dict(cell_object=cellObject, target_segs=target_segments) 

                for new_target_group in new_seg_length_dict.keys():

                    cached_dicts[cell_component]['TargetDict'][new_target_group] = new_seg_length_dict[new_target_group]

                    segLengthDict[new_target_group] = new_seg_length_dict[new_target_group]

    else:

        cell_nml_file = '%s.cell.nml' % cell_component

        if path_to_nml2 != None:

            document_cell = neuroml.loaders.NeuroMLLoader.load(os.path.join(path_to_nml2, cell_nml_file))

        else:

            document_cell = neuroml.loaders.NeuroMLLoader.load(cell_nml_file)

        cellObject = document_cell.cells[0]

        target_segments = oc_build.extract_seg_ids(cell_object=cellObject, target_compartment_array=list_of_target_seg_groups, targeting_mode='segGroups')

        segLengthDict = oc_build.make_target_dict(cell_object=cellObject, target_segs=target_segments) 

        cached_dicts[cell_component] = {}

        cached_dicts[cell_component]['CellObject'] = cellObject

        cached_dicts[cell_component]['TargetDict'] = segLengthDict

    return segLengthDict, cached_dicts


##############################################################################################

def check_pre_segment_groups(pre_segment_group_info):

    error_counter = 0

    passed = False

    if len(pre_segment_group_info) == 1:

        if isinstance(pre_segment_group_info[0], dict):

            check_pre_seg = 'PreSegGroup' in pre_segment_group_info[0].keys()

            check_proj_type = 'ProjType' in pre_segment_group_info[0].keys()     

            if not check_pre_seg:

                opencortex.print_comment_v("Error in build connectivity: the key 'PreSegGroup' is not in the dictionary keys inside pre_segment_group_info.")
                error_counter += 1

            else:

                if not isinstance(pre_segment_group_info[0]['PreSegGroup'], str):

                    opencortex.print_comment_v("Error in build connectivity: the value of the key 'PreSegGroup' in the dictionary keys inside pre_segment_group_info"
                                               " must be of type 'str'. Only one presynaptic segment group is allowed per projection.")
                    error_counter += 1 

            if not check_proj_type:

                opencortex.print_comment_v("Error in build connectivity: the key 'ProjType' is not in the dictionary keys inside pre_segment_group_info.")
                error_counter += 1

        else:

            opencortex.print_comment_v("Error in build_connectivity: the list pre_segment_group_info has only one element but it is not of type 'dict'."
                                       " The current type is %s." % (type(pre_segment_group_info[0])))
            error_counter += 1

    if len(pre_segment_group_info) > 1:

        for proj in range(0, len(pre_segment_group_info)):

            if isinstance(pre_segment_group_info[proj], dict):

                check_pre_seg = 'PreSegGroup' in pre_segment_group_info[proj].keys()

                check_pre_pop = 'PrePop' in pre_segment_group_info[proj].keys()

                check_post_pop = 'PostPop' in pre_segment_group_info[proj].keys()

                check_proj_type = 'ProjType' in pre_segment_group_info[proj].keys()

                if not check_pre_seg:

                    opencortex.print_comment_v("Error in build connectivity: the key 'PreSegGroup' is not in the dictionary keys inside pre_segment_group_info.")
                    error_counter += 1

                else:

                    if not isinstance(pre_segment_group_info[proj]['PreSegGroup'], str):

                        opencortex.print_comment_v("Error in build connectivity: the value of the key 'PreSegGroup' in the dictionary keys inside pre_segment_group_info"
                                                   " must be of type 'str'. Only one presynaptic segment group is allowed per projection.")
                        error_counter += 1

                if not check_pre_pop:

                    opencortex.print_comment_v("Error in build connectivity: the key 'PrePop' is not in the dictionary keys inside pre_segment_group_info.")
                    error_counter += 1

                if not check_post_pop:

                    opencortex.print_comment_v("Error in build connectivity: the key 'PostPop' is not in the dictionary keys inside pre_segment_group_info.")
                    error_counter += 1

                if not check_proj_type:

                    opencortex.print_comment_v("Error in build connectivity: the key 'ProjType' is not in the dictionary keys inside pre_segment_group_info.")
                    error_counter += 1

            else:

                opencortex.print_comment_v("Error in build_connectivity: the list elements in the pre_segment_group_info must be dictionaries with fields"
                                           " 'PreSegGroup', 'PrePop', 'PostPop' and 'ProjType'. The current type is %s." % (type(pre_segment_group_info[proj])))
                error_counter += 1

    if error_counter == 0:

        passed = True

    return passed


##############################################################################################

def build_inputs(nml_doc, net, population_params, input_params, cached_dicts=None, path_to_cells=None, path_to_synapses=None):     

    '''
    a wrapper method that calls appropriate methods to build inputs to the NeuroML2 network. Input arguments:

    nml_doc - a libNeuroML doc object;

    net - a libNeuroML net object;

    population_params - a dictionary that stores population parameters in the format returned by the method add_populations_in_layers;

    input_params -a dictionary that specifies input parameters for any given cell model. The format can be checked by the method check_inputs. Dictionary values must

    be of type 'list' and can thus define multiple input groups on any given cell type. Examples where lists contain only one input group but differ in other parameters:

    Example 1: input_params={'TCR':[{'InputType':'GeneratePoissonTrains',
                  'InputName':'TransPoiInputs',
                  'TrainType':'transient',
                  'Synapse':'Syn_AMPA_L6NT_TCR',
                  'AverageRateList':[0.05],
                  'DurationList':[200.0],
                  'DelayList':[20.0],
                  'FractionToTarget':1.0,
                  'LocationSpecific':False,
                  'TargetRegions':[{'XVector':[2,12],'YVector':[3,5],'ZVector':[0,5]}],
                  'TargetDict':{'dendrite_group':1000 }       }]              }     

    Example 2: input_params={'TCR':[{'InputType':'PulseGenerators',
                     'InputName':'PulseGenerator0',
                     'Noise':False,
                     'AmplitudeList':[20.0,-20.0],
                     'DurationList':[100.0,50.0],
                     'DelayList':[50.0,200.0],
                     'FractionToTarget':1.0,
                     'LocationSpecific':False,
                     'TargetDict':{'dendrite_group':2 }       }]              }   

    Example 3: input_params={'CG3D_L23PyrRS':[{'InputType':'PulseGenerators',
                             'InputName':'PulseGenerator0',
                             'Noise':True,
                             'SmallestAmplitudeList':[5.0E-5],
                             'LargestAmplitudeList':[1.0E-4],
                             'DurationList':[20000.0],
                             'DelayList':[0.0],
                             'TimeUnits':'ms',
                             'AmplitudeUnits':'uA',
                             'FractionToTarget':1.0,
                             'LocationSpecific':False,
                             'UniqueTargetSegmentID':0,
                             'UniqueFractionAlong':0.5  }]             }                            ;

    cached_dicts - optional argument, default value is set to None; otherwise should be the input variable that stores the dictionary in the format returned by check_cached_dicts;

    path_to_cells - dir where NeuroML2 cell files are found;

    path_to_synapses - dir where NeuroML2 synapse files are found.          '''

    passed_inputs = check_inputs(input_params, population_params, path_to_cells, path_to_synapses)

    if passed_inputs:

        opencortex.print_comment_v("Input parameters were specified correctly.")

    else:

        opencortex.print_comment_v("Input parameters were specified incorrectly; execution will terminate.")

        quit()

    input_list_array_final = []        

    input_synapse_list = []                    

    for cell_tag in input_params.keys():

        for cell_population in population_params.keys():

            if cell_tag in cell_population:

                pop = population_params[cell_population]['PopObj']

                cell_component = pop.component

                for input_group_ind in range(0, len(input_params[cell_tag])):

                    input_group_params = input_params[cell_tag][input_group_ind]

                    input_group_tag = input_group_params['InputName'] + "_TO_" + pop.id

                    popID = cell_population

                    fraction_to_target = input_group_params['FractionToTarget']

                    if not input_group_params['LocationSpecific']:

                        target_cell_ids = oc_build.get_target_cells(population=pop, fraction_to_target=fraction_to_target)

                    else:

                        list_of_regions = input_group_params['TargetRegions']

                        x_list = []

                        y_list = []

                        z_list = []

                        for region_index in range(0, len(list_of_regions)):

                            x_list.append(list_of_regions[region_index]['XVector'])

                            y_list.append(list_of_regions[region_index]['YVector'])

                            z_list.append(list_of_regions[region_index]['ZVector'])

                        target_cell_ids = oc_build.get_target_cells(population=pop,
                                                                    fraction_to_target=fraction_to_target, 
                                                                    list_of_xvectors=x_list,
                                                                    list_of_yvectors=y_list,
                                                                    list_of_zvectors=z_list)

                    if target_cell_ids != []:

                        input_ids_final = []

                        weight_list_final = []

                        condition1 = 'TargetDict' in input_group_params.keys()

                        condition2 = 'UniversalTargetSegmentID' not in input_group_params.keys()

                        condition3 = 'UniversalFractionAlong' not in input_group_params.keys()
                        
                        weight_dict = {}

                        if condition1 and condition2 and condition3:

                            subset_dict = input_group_params['TargetDict']

                            target_segment = None

                            fraction_along = None

                            if None not in input_group_params['TargetDict'].keys():

                                target_group_list = subset_dict.keys()

                                if cached_dicts != None:

                                    segLengthDict, cached_dicts = check_cached_dicts(cell_component, cached_dicts, target_group_list, path_to_nml2=path_to_cells)

                                else:

                                    target_segments = oc_build.extract_seg_ids(cell_object=cellObject,
                                                                               target_compartment_array=input_group_params['TargetDict'].keys(),
                                                                               targeting_mode='segGroups')

                                    segLengthDict = oc_build.make_target_dict(cell_object=cellObject, target_segs=target_segments) 

                            else:

                                segLengthDict = None

                        else:

                            target_segment = input_group_params['UniversalTargetSegmentID']

                            fraction_along = input_group_params['UniversalFractionAlong']

                            segLengthDict = None

                            subset_dict = None

                        if input_group_params['InputType'] == 'GeneratePoissonTrains':

                            list_of_input_ids = []

                            if input_group_params['TrainType'] == 'transient':

                                for input_index in range(0, len(input_group_params['AverageRateList'])):

                                    tpfs = oc_build._add_transient_poisson_firing_synapse(nml_doc=nml_doc, 
                                                                                         id=input_group_tag + "_TransPoiSyn%d" % input_index, 
                                                                                         average_rate="%f %s" % (input_group_params['AverageRateList'][input_index], input_group_params['RateUnits']),
                                                                                         delay="%f %s" % (input_group_params['DelayList'][input_index], input_group_params['TimeUnits']),
                                                                                         duration="%f %s" % (input_group_params['DurationList'][input_index], input_group_params['TimeUnits']), 
                                                                                         synapse_id=input_group_params['Synapse'])

                                    input_synapse_list.append(input_group_params['Synapse'])                                          
                                    list_of_input_ids.append(tpfs.id)

                            if input_group_params['TrainType'] == 'persistent':

                                for input_index in range(0, len(input_group_params['AverageRateList'])):

                                    pfs = oc_build._add_poisson_firing_synapse(nml_doc=nml_doc, 
                                                                              id=input_group_tag + "_PoiSyn%d" % input_index, 
                                                                              average_rate="%f %s" % (input_group_params['AverageRateList'][input_index], input_group_params['RateUnits']), 
                                                                              synapse_id=input_group_params['Synapse'])

                                    input_synapse_list.append(input_group_params['Synapse'])                           
                                    list_of_input_ids.append(pfs.id)

                            input_ids_final.append(list_of_input_ids)

                        if input_group_params['InputType'] == 'PulseGenerators':

                            if not input_group_params['Noise']:

                                list_of_input_ids = []

                                for input_index in range(0, len(input_group_params['AmplitudeList'])):

                                    pg = oc_build._add_pulse_generator(nml_doc=nml_doc, 
                                                                      id=input_group_tag + "_Pulse%d" % input_index, 
                                                                      delay="%f %s" % (input_group_params['DelayList'][input_index], input_group_params['TimeUnits']),
                                                                      duration="%f %s" % (input_group_params['DurationList'][input_index], input_group_params['TimeUnits']), 
                                                                      amplitude="%f %s" % (input_group_params['AmplitudeList'][input_index], input_group_params['AmplitudeUnits']))

                                    list_of_input_ids.append(pg.id)

                                input_ids_final.append(list_of_input_ids)

                            else:

                                for cell_id in target_cell_ids:
                            
                                    assert len(input_group_params['SmallestAmplitudeList'])==1 
                                    
                                    for input_index in range(0, len(input_group_params['SmallestAmplitudeList'])):

                                        random_amplitude = random.uniform(input_group_params['SmallestAmplitudeList'][input_index], input_group_params['LargestAmplitudeList'][input_index])
                                    
                                        weight_dict[cell_id] = random_amplitude
                                        
                                list_of_input_ids = []

                                pg = oc_build._add_pulse_generator(nml_doc=nml_doc, 
                                                                  id=input_group_tag + "_Pulse", 
                                                                  delay="%s %s" % (input_group_params['DelayList'][0], input_group_params['TimeUnits']),
                                                                  duration="%s %s" % (input_group_params['DurationList'][0], input_group_params['TimeUnits']), 
                                                                  amplitude="%s %s" % (1, input_group_params['AmplitudeUnits']))

                                list_of_input_ids.append(pg.id)

                                input_ids_final.append(list_of_input_ids)

                        if input_group_params['InputType'] == 'GenerateSpikeSourcePoisson':

                            list_of_input_ids = []     

                            weight_list = [] 

                            for input_index in range(0, len(input_group_params['AverageRateList'])):

                                ssp = oc_build._add_spike_source_poisson(nml_doc=nml_doc, 
                                                                        id=input_group_tag + "_SpSourcePoi%d" % input_index, 
                                                                        start="%f %s" % (input_group_params['DelayList'][input_index], input_group_params['TimeUnits']),
                                                                        duration="%f %s" % (input_group_params['DurationList'][input_index], input_group_params['TimeUnits']),
                                                                        rate="%f %s" % (input_group_params['AverageRateList'][input_index], input_group_params['RateUnits']))

                                input_synapse_list.append(input_group_params['Synapse'])

                                list_of_input_ids.append(ssp.id)

                                weight_list.append(input_group_params['WeightList'][input_index])

                            input_ids_final.append(list_of_input_ids)

                            weight_list_final.append(weight_list)

                        if input_group_params['InputType'] == 'GeneratePoissonTrains' or input_group_params['InputType'] == 'PulseGenerators':

                            input_list_array = oc_build.add_advanced_inputs_to_population(net=net, 
                                                                                          id=input_group_tag, 
                                                                                          population=pop,
                                                                                          input_id_list=input_ids_final,
                                                                                          seg_length_dict=segLengthDict,
                                                                                          subset_dict=subset_dict,
                                                                                          universal_target_segment=target_segment,
                                                                                          universal_fraction_along=fraction_along,
                                                                                          only_cells=target_cell_ids,
                                                                                          weight_dict=weight_dict)

                        if input_group_params['InputType'] == 'GenerateSpikeSourcePoisson':

                            input_list_array = oc_build.add_projection_based_inputs(net=net,
                                                                                    id=input_group_tag,
                                                                                    population=pop,
                                                                                    input_id_list=input_ids_final,
                                                                                    weight_list=weight_list_final,
                                                                                    synapse_id=input_group_params['Synapse'],
                                                                                    seg_length_dict=segLengthDict,
                                                                                    subset_dict=subset_dict,
                                                                                    universal_target_segment=target_segment,
                                                                                    universal_fraction_along=fraction_along,
                                                                                    only_cells=target_cell_ids)


                        input_list_array_final.append(input_list_array)

    input_synapse_list = list(set(input_synapse_list))  

    return input_list_array_final, input_synapse_list  


##############################################################################################

def replace_cell_types(net_file_name,
                       path_to_net,
                       new_net_id,
                       cell_types_to_be_replaced,
                       cell_types_replaced_by,
                       dir_to_new_components,
                       dir_to_old_components,
                       reduced_to_single_compartment=True,
                       validate_nml2=True,
                       return_synapses=False,
                       connection_segment_groups=None,
                       input_segment_groups=None,
                       synapse_file_tags=None):

    '''This method substitutes the target cell types to a given NeuroML2 cortical network. '''

    if len(cell_types_to_be_replaced) == len(cell_types_replaced_by):

        nml2_file_path = os.path.join(path_to_net, net_file_name + ".net.nml")      

        net_doc = pynml.read_neuroml2_file(nml2_file_path)

        net_doc.id = new_net_id

        include_synapses = []

        list_of_synapses = []

        if synapse_file_tags != None:

            for include_index in range(0, len(net_doc.includes)):

                found_synapse = False

                for synapse_tag in synapse_file_tags:

                    if synapse_tag in net_doc.includes[include_index].href:

                        found_synapse = True

                        break

                if found_synapse:

                    include_synapses.append(net_doc.includes[include_index])

                    list_of_synapses.append(net_doc.includes[include_index].href)

        net_doc.includes = []

        for syn_index in range(0, len(include_synapses)):

            net_doc.includes.append(include_synapses[syn_index])            

        if (not reduced_to_single_compartment) and (connection_segment_groups != None or input_segment_groups != None):

            cached_target_dict = {}

        net = net_doc.networks[0]

        net.id = new_net_id

        old_to_new = []

        all_old_components = []

        for pop_counter in range(0, len(net.populations)):

            pop = net.populations[pop_counter]

            all_old_components.append(pop.component)

            for cell_index in range(0, len(cell_types_replaced_by)):

                if pop.component == cell_types_to_be_replaced[cell_index]:

                    conversion_dict = {}

                    conversion_dict['OldPopID'] = pop.id

                    conversion_dict['OldCellComponent'] = pop.component

                    pop.component = cell_types_replaced_by[cell_index]

                    if cell_types_to_be_replaced[cell_index] in pop.id:

                        pop.id = pop.id.replace(cell_types_to_be_replaced[cell_index], cell_types_replaced_by[cell_index])

                    conversion_dict['NewPopID'] = pop.id

                    conversion_dict['NewCellComponent'] = pop.component

                    old_to_new.append(conversion_dict)

        all_old_components = list(set(all_old_components))

        all_projections_final = []

        if hasattr(net, 'projections'):

            chemical_projections = {}

            chemical_projections['Type'] = 'Chem'

            chemical_projections['Projs'] = net.projections

            all_projections_final.append(chemical_projections)

        if hasattr(net, 'electrical_projections'):

            electrical_projections = {}

            electrical_projections['Type'] = 'Elect'

            electrical_projections['Projs'] = net.electrical_projections

            all_projections_final.append(electrical_projections)

        for proj_group_index in range(0, len(all_projections_final)):

            proj_dict = all_projections_final[proj_group_index]

            projections = proj_dict['Projs']

            proj_type = proj_dict['Type']

            for proj_counter in range(0, len(projections)):

                proj = projections[proj_counter]

                replaced_pre_pop = False

                replaced_post_pop = False

                for conversion_index in range(0, len(old_to_new)):

                    conversion_params = old_to_new[conversion_index]

                    if proj.presynaptic_population == conversion_params['OldPopID']:

                        replaced_pre_pop = True

                        pre_pop_index = conversion_index

                        if proj.presynaptic_population in proj.id:

                            proj.id = proj.id.replace(proj.presynaptic_population, conversion_params['NewPopID'])

                        proj.presynaptic_population = conversion_params['NewPopID']

                    if proj.postsynaptic_population == conversion_params['OldPopID']:

                        replaced_post_pop = True

                        post_pop_index = conversion_index

                        if proj.postsynaptic_population in proj.id:

                            proj.id = proj.id.replace(proj.postsynaptic_population, conversion_params['NewPopID'])

                        proj.postsynaptic_population = conversion_params['NewPopID']

                pre_seg_length_dict = None

                pre_subset_dict = None

                post_seg_length_dict = None

                post_subset_dict = None

                if (not reduced_to_single_compartment) and connection_segment_groups != None:

                    if replaced_pre_pop and replaced_post_pop:

                        for index in range(0, len(connection_segment_groups)):

                            proj_info = connection_segment_groups[index]

                            check_pre_cell_type = old_to_new[pre_pop_index]['NewCellComponent'] == proj_info['PreCellType']

                            check_post_cell_type = old_to_new[post_pop_index]['NewCellComponent'] == proj_info['PostCellType']

                            check_proj_type = proj_info['Type'] == proj_type

                            if check_pre_cell_type and check_post_cell_type and check_proj_type:

                                pre_seg_length_dict, cached_target_dict = check_cached_dicts(old_to_new[pre_pop_index]['NewCellComponent'],
                                                                                             cached_target_dict,
                                                                                             [proj_info['PreSegGroup']],
                                                                                             path_to_nml2=dir_to_new_components) 

                                pre_subset_dict = {}

                                pre_subset_dict[proj_info['PreSegGroup']] = 1

                                post_seg_length_dict, cached_target_dict = check_cached_dicts(old_to_new[post_pop_index]['NewCellComponent'],
                                                                                              cached_target_dict,
                                                                                              [proj_info['PostSegGroup']],
                                                                                              path_to_nml2=dir_to_new_components) 

                                post_subset_dict = {}

                                post_subset_dict[proj_info['PostSegGroup']] = 1

                                break


                if hasattr(proj, 'connection_wds'):

                    if proj.connection_wds != []:

                        connections = proj.connection_wds

                        id_tag = True

                elif  hasattr(proj, 'connections'):

                    if proj.connections != []:

                        connections = proj.connections

                        id_tag = True

                elif hasattr(proj, 'electrical_connection_instances'):

                    if proj.electrical_connection_instances != []:

                        connections = proj.electrical_connection_instances

                        id_tag = False

                else:

                    if proj.electrical_connections != []:

                        connections = proj.electrical_connections

                        id_tag = False

                for conn_counter in range(0, len(connections)):

                    connection = connections[conn_counter]

                    if replaced_post_pop:

                        if id_tag:

                            if old_to_new[post_pop_index]['OldCellComponent'] in connection.post_cell_id:

                                connection.post_cell_id = connection.post_cell_id.replace(old_to_new[post_pop_index]['OldCellComponent'], old_to_new[post_pop_index]['NewCellComponent'])

                            if old_to_new[post_pop_index]['OldPopID'] in connection.post_cell_id:

                                connection.post_cell_id = connection.post_cell_id.replace(old_to_new[post_pop_index]['OldPopID'], old_to_new[post_pop_index]['NewPopID'])

                        else:

                            if old_to_new[post_pop_index]['OldCellComponent'] in connection.post_cell:

                                connection.post_cell = connection.post_cell.replace(old_to_new[post_pop_index]['OldCellComponent'], old_to_new[post_pop_index]['NewCellComponent'])

                            if old_to_new[post_pop_index]['OldPopID'] in connection.post_cell:

                                connection.post_cell = connection.post_cell.replace(old_to_new[post_pop_index]['OldPopID'], old_to_new[post_pop_index]['NewPopID'])

                        if post_seg_length_dict != None and post_subset_dict != None:

                            post_target_seg_array, post_target_fractions = oc_build.get_target_segments(post_seg_length_dict, post_subset_dict)

                            if id_tag:

                                connection.post_segment_id = post_target_seg_array[0]

                            else:

                                connection.post_segment = post_target_seg_array[0]

                            connection.post_fraction_along = post_target_fractions[0]

                        else:

                            if reduced_to_single_compartment:

                                if id_tag:

                                    connection.post_segment_id = 0

                                else:

                                    connection.post_segment = 0

                                connection.post_fraction_along = 0.5

                    if replaced_pre_pop:

                        if id_tag:

                            if old_to_new[pre_pop_index]['OldCellComponent'] in connection.pre_cell_id:

                                connection.pre_cell_id = connection.pre_cell_id.replace(old_to_new[pre_pop_index]['OldCellComponent'], old_to_new[pre_pop_index]['NewCellComponent'])

                            if old_to_new[pre_pop_index]['OldPopID'] in connection.pre_cell_id:

                                connection.pre_cell_id = connection.pre_cell_id.replace(old_to_new[pre_pop_index]['OldPopID'], old_to_new[pre_pop_index]['NewPopID'])
                        else:

                            if old_to_new[pre_pop_index]['OldCellComponent'] in connection.pre_cell:

                                connection.pre_cell = connection.pre_cell.replace(old_to_new[pre_pop_index]['OldCellComponent'], old_to_new[pre_pop_index]['NewCellComponent'])

                            if old_to_new[pre_pop_index]['OldPopID'] in connection.pre_cell:

                                connection.pre_cell = connection.pre_cell.replace(old_to_new[pre_pop_index]['OldPopID'], old_to_new[pre_pop_index]['NewPopID'])

                        if pre_seg_length_dict != None and pre_subset_dict != None:

                            pre_target_seg_array, pre_target_fractions = oc_build.get_target_segments(pre_seg_length_dict, pre_subset_dict)

                            if id_tag:

                                connection.pre_segment_id = pre_target_seg_array[0]

                            else:

                                connection.pre_segment = pre_target_seg_array[0]

                            connection.pre_fraction_along = pre_target_fractions[0]

                        else:

                            if reduced_to_single_compartment:

                                if id_tag:

                                    connection.pre_segment_id = 0

                                else:

                                    connection.pre_segment = 0

                                connection.pre_fraction_along = 0.5

        for input_list_index in range(0, len(net.input_lists)):

            input_list_obj = net.input_lists[input_list_index]

            for conversion_index in range(0, len(old_to_new)):

                conversion_params = old_to_new[conversion_index]

                if conversion_params['OldPopID'] == input_list_obj.populations:

                    input_list_obj.populations = conversion_params['NewPopID']

                    require_segment_groups = False

                    if (not reduced_to_single_compartment) and input_segment_groups != None:

                        for index in range(0, len(input_segment_groups)):

                            input_group_info = input_segment_groups[index]

                            check_post_cell_type = conversion_params['NewCellComponent'] == input_group_info['PostCellType']

                            if check_post_cell_type:

                                target_segment_dict, cached_target_dict = check_cached_dicts(conversion_params['NewCellComponent'],
                                                                                             cached_target_dict,
                                                                                             [input_group_info['PostSegGroup']],
                                                                                             path_to_nml2=dir_to_new_components) 

                                target_subset_dict = {}

                                target_subset_dict[input_group_info['PostSegGroup']] = 1

                                require_segment_groups = True

                                break

                    for input_index in range(0, len(input_list_obj.input)):

                        input_obj = input_list_obj.input[input_index]

                        if conversion_params['OldPopID'] in input_obj.target:

                            input_obj.target = input_obj.target.replace(conversion_params['OldPopID'], conversion_params['NewPopID'])

                        if conversion_params['OldCellComponent'] in input_obj.target:

                            input_obj.target = input_obj.target.replace(conversion_params['OldCellComponent'], conversion_params['NewCellComponent'])

                        if require_segment_groups:

                            if input_obj.segment_id != 0:

                                target_seg_array, target_fractions = oc_build.get_target_segments(target_segment_dict, target_subset_dict)

                                input_obj.segment_id = target_seg_array[0]

                                input_obj.fraction_along = target_fractions[0]

                        else:

                            input_obj.segment_id = 0

                            input_obj.fraction_along = 0.5


                    break 

        replaced_components_final = []

        for changed_cell in range(0, len(old_to_new)):

            cell_model = old_to_new[changed_cell]['NewCellComponent']

            replaced_components_final.append(old_to_new[changed_cell]['OldCellComponent'])

            oc_build.add_cell_and_channels(net_doc, os.path.join(dir_to_new_components, "%s.cell.nml" % cell_model), cell_model, use_prototypes=False)

        for cell_model in all_old_components:

            if cell_model not in replaced_components_final:

                oc_build.add_cell_and_channels(net_doc, os.path.join(dir_to_old_components, "%s.cell.nml" % cell_model), cell_model, use_prototypes=False)

        nml_file_name = "%s.net.nml" % new_net_id

        oc_build.save_network(net_doc, nml_file_name, validate=validate_nml2)

        if return_synapses:

            return list_of_synapses

    else:

        opencortex.print_comment_v("Error: the number of cell types in the list cell_types_to_be_replaced is not equal to the number of new cell types in the list"
                                   " cell_types_replaced_by.") 

        quit()


##############################################################################################

def parse_distance_dependence_params(distance_dependence_params, pre_pop, post_pop, proj_type):

    dist_rule = None

    for distance_param in range(0, len(distance_dependence_params)):

        check_pre_pop = distance_dependence_params[distance_param]['PrePopID'] == pre_pop

        check_post_pop = distance_dependence_params[distance_param]['PostPopID'] == post_pop

        check_proj_type = distance_dependence_params[distance_param]['Type'] == proj_type

        if check_pre_pop and check_post_pop and check_proj_type:

            dist_rule = distance_dependence_params['DistDependConn']

            break

    return dist_rule


##############################################################################################

def parse_delays(delays_params, post_pop, synapse_list):

    delays = {}

    for syn_ind in range(0, len(synapse_list)):

        for delay_param in range(0, len(delays_params)):

            if delays_params[delay_param]['synComp'] == 'all':

                delays[synapse_list[syn_ind]] = float(delays_params[delay_param]['delay'])

            else:

                passed_synComp = False

                if delays_params[delay_param]['synComp'] in synapse_list[syn_ind]:

                    passed_synComp = True

                passed_synEndsWith = False

                if delays_params[delay_param]['synEndsWith'] == []:

                    passed_synEndsWith = True

                else:

                    for syn_end in delays_params[delay_param]['synEndsWith']:

                        if synapse_list[syn_ind].endswith(syn_end):

                            passed_synEndsWith = True

                            break

                passed_targetCellGroup = False

                if delays_params[delay_param]['targetCellGroup'] == []:

                    passed_targetCellGroup = True

                else:

                    for target_cell_group in delays_params[delay_param]['targetCellGroup']:

                        if target_cell_group in post_pop:

                            passed_targetCellGroup = True

                            break

                if passed_synComp and passed_synEndsWith and passed_targetCellGroup:

                    delays[synapse_list[syn_ind]] = float(delays_params[delay_param]['delay'])


    if delays.keys() == []:

        delays = None

    return delays


##############################################################################################

def parse_weights(weights_params, post_pop, synapse_list):

    weights = {}

    for syn_ind in range(0, len(synapse_list)):

        for weight_param in range(0, len(weights_params)):

            if weights_params[weight_param]['synComp'] == 'all':

                weights[synapse_list[syn_ind]] = float(weights_params[weight_param]['weight'])

            else:

                passed_synComp = False

                if weights_params[weight_param]['synComp'] in synapse_list[syn_ind]:

                    passed_synComp = True

                passed_synEndsWith = False

                if weights_params[weight_param]['synEndsWith'] == []:

                    passed_synEndsWith = True

                else:

                    for syn_end in weights_params[weight_param]['synEndsWith']:

                        if synapse_list[syn_ind].endswith(syn_end):

                            passed_synEndsWith = True

                            break

                passed_targetCellGroup = False

                if weights_params[weight_param]['targetCellGroup'] == []:

                    passed_targetCellGroup = True

                else:

                    for target_cell_group in weights_params[weight_param]['targetCellGroup']:

                        if target_cell_group in post_pop:

                            passed_targetCellGroup = True

                            break

                if passed_synComp and passed_synEndsWith and passed_targetCellGroup:

                    weights[synapse_list[syn_ind]] = float(weights_params[weight_param]['weight'])


    if weights.keys() == []:

        weights = None

    return weights


##############################################################################################

def check_includes_in_cells(dir_to_cells,
                            list_of_cell_ids,
                            extra_channel_tags=None):

    passed = True

    list_of_cell_file_names = []

    for cell_id in list_of_cell_ids:

        list_of_cell_file_names.append(cell_id + ".cell.nml")

    all_src_files = os.listdir(dir_to_cells)

    target_files = list_of_cell_file_names

    for src_file in all_src_files:

        if src_file not in target_files:

            target_files.append(src_file)

    for cell_file_name in target_files:

        full_path_to_cell = os.path.join(dir_to_cells, cell_file_name)

        if not os.path.exists(full_path_to_cell):

            passed = False

            opencortex.print_comment_v("Error: path %s does not exist; use method copy_nml2_source to copy nml2 files from the source directory to the appropriate NeuroML2 component directories." % full_path_to_cell)

            break

        else:

            nml2_doc_cell = pynml.read_neuroml2_file(full_path_to_cell, include_includes=False)

            for included in nml2_doc_cell.includes:

                if '.channel.nml' in included.href:

                    if ('../channels/' not in included.href) or ('..\channels\'' not in included.href):

                        channel_dir = os.path.join("..", "channels")

                        included.href = os.path.join(channel_dir, included.href)

                        continue

                else:

                    if extra_channel_tags != None:

                        for channel_tag in included.href:

                            if channel_tag in included.href:

                                if ('../channels/' not in included.href) or ('..\channels\'' not in included.href):

                                    channel_dir = os.path.join("..", "channels")

                                    included.href = os.path.join(channel_dir, included.href)

                                    break

            pynml.write_neuroml2_file(nml2_doc_cell, full_path_to_cell)    

    return passed


##############################################################################################

def check_pop_dict_and_layers(pop_dict, boundary_dict):

    error_counter = 0

    passed = False

    for cell_population in pop_dict.keys():

        if not isinstance(pop_dict[cell_population], tuple):

            print("TypeError in population parameters: the values stored in the population dictionary must be tuples.")
            print("The current type is %s" % (type(pop_dict[cell_population])))
            error_counter += 1

        else:

            if len(pop_dict[cell_population]) != 5:

                print("ValueError in population parameters: tuples in the population dictionary must contain four elements in the following order and type: "
                      "population size ('int'), layer ('str'), cell type ('str'), compartmentalization type ('single' or 'multi'), color ('str', e.g. '1 0 0' for red).")
                error_counter += 1

            else:

                if not isinstance(pop_dict[cell_population][0], int):

                    print("TypeError in population parameters: the first element in tuples in the population dictionary must be of type 'int'")
                    print(" as it specifies the size of cell population. The current type of the first element is %s" % (type(pop_dict[cell_population][0])))
                    error_counter += 1

                if not isinstance(pop_dict[cell_population][1], str):

                    print("TypeError in population parameters: the second element in tuples in the population dictionary must be of type 'string'")
                    print(" as it specifies the layer of cell population. The current type of the second element is %s" % (type(pop_dict[cell_population][1])))
                    error_counter += 1

                else:

                    try:

                        test_layer = boundary_dict[pop_dict[cell_population][1]]

                    except KeyError:

                        print("KeyError in the layer boundary dictionary: cell population id '%s' is not in the keys of the layer boundary dictionary" % cell_population)
                        error_counter += 1

                if not isinstance(pop_dict[cell_population][2], str):

                    print("TypeError in population parameters: the third element in tuples in the population dictionary must be of type 'string'")
                    print(" as it specifies the cell model for a given population. The current type of the third element is %s" % (type(pp_dict[cell_population][2])))
                    error_counter += 1

                if not isinstance(pop_dict[cell_population][3], str):

                    print("TypeError in population parameters: the fourth element in tuples in the population dictionary must be of type 'string'")
                    print(" as it specifies the compartmentalization for a given cell type. The current type of the fourth element is %s" % (type(pp_dict[cell_population][2])))
                    error_counter += 1

                else:

                    if not (pop_dict[cell_population][3] == 'single' or pop_dict[cell_population][3] == 'multi'):

                        print("ValueError in population parameters: the fourth element in tuples in the population dictionary must be equal to 'single' or 'multi'"
                              " as it specifies the compartmentalization for a given cell type. The current type of the fourth element is %s")
                        error_counter += 1

    if error_counter == 0:

        passed = True       


    return passed


##############################################################################################

def check_synapse_location(synapse_id, pathToSynapses):

    found = False

    src_files = os.listdir(pathToSynapses)

    for file_name in src_files:
        if synapse_id in file_name:
            found = True   

    return found  


##############################################################################################

def get_segment_groups(cell_id, path_to_cells):

    if path_to_cells != None:

        cell_nml_file = os.path.join(path_to_cells, '%s.cell.nml' % cell_id)

    else:

        cell_nml_file = '%s.cell.nml' % cell_id

    document_cell = neuroml.loaders.NeuroMLLoader.load(cell_nml_file)

    cell_object = document_cell.cells[0]

    segment_groups = []

    for segment_group in cell_object.morphology.segment_groups:

        segment_groups.append(segment_group.id)

    return segment_groups


##############################################################################################

def check_segment_group(segment_groups, target_segment_group):
    segment_group_exist = False
    if target_segment_group in segment_groups:
        segment_group_exist = True
    return segment_group_exist


##############################################################################################

def check_weight_params(weight_params):

    error_counter = 0

    if not isinstance(weight_params, list):

        print("TypeError in weight parameters: weight parameters must be of type 'list'. The current type is '%s'." % (type(weight_params)))

        error_counter += 1

    else:

        for weight_param in range(0, len(weight_params)):

            if not isinstance(weight_params[weight_param], dict):

                print("TypeError in weight parameters: list elements in weight parameters must be of type 'dict'. The current type is '%s'." % (type(weight_params[weight_param])))

                error_counter += 1

            else:

                try:

                    test_weight_field = weight_params[weight_param]['weight']

                except KeyError:

                    print("KeyError in weight parameters: the key 'weight' is not in the keys of weight parameter dictionary.")

                    error_counter += 1

                try:

                    test_syn_comp_field = weight_params[weight_param]['synComp']

                    if not isinstance(test_syn_comp_field, str):

                        print("TypeError in weight parameters: the value of the key 'synComp' must be of type 'str'. The current type is '%s'." % (type(test_syn_comp_field)))

                        error_counter += 1

                    else:

                        if test_syn_comp_field != 'all':

                            try:

                                test_syn_ends_with_field = weight_params[weight_param]['synEndsWith']

                                if not isinstance(test_syn_ends_with_field, list):

                                    print("TypeError in weight parameters: the value of the key 'synEndsWith' must be of type 'list'.")
                                    print("The current type is '%s'." % (type(test_syn_ends_with_field)))

                                    error_counter += 1

                            except KeyError:

                                print("KeyError in weight parameters: the key 'synEndsWith' is not in the keys of weight parameter dictionary.")

                                error_counter += 1

                            try:

                                test_target_cell_group = weight_params[weight_param]['targetCellGroup']

                                if not isinstance(test_target_cell_group, list):

                                    print("TypeError in weight parameters: the value of the key 'targetCellGroup' must be of type 'list'.")
                                    print("The current type is '%s'." % (type(test_target_cell_group)))

                                    error_counter += 1

                            except KeyError:

                                print("KeyError in weight parameters: the key 'targetCellGroup' is not in the keys of weight parameter dictionary.")

                                error_counter += 1

                except KeyError:

                    print("KeyError in weight parameters: the key 'synComp' is not in the keys of weight parameter dictionary.")

                    error_counter += 1

    if error_counter == 0:

        return True

    else:

        return False              


##############################################################################################

def check_delay_params(delay_params):

    error_counter = 0

    if not isinstance(delay_params, list):

        print("TypeError in delay parameters: delay parameters must be of type 'list'. The current type is '%s'." % (type(delay_params)))

        error_counter += 1

    else:

        for delay_param in range(0, len(delay_params)):

            if not isinstance(delay_params[delay_param], dict):

                print("TypeError in delay parameters: list elements in delay parameters must be of type 'dict'. The current type is '%s'." % (type(delay_params[delay_param])))

                error_counter += 1

            else:

                try:

                    test_weight_field = delay_params[delay_param]['delay']

                except KeyError:

                    print("KeyError in delay parameters: the key 'delay' is not in the keys of delay parameter dictionary.")

                    error_counter += 1

                try:

                    test_syn_comp_field = delay_params[delay_param]['synComp']

                    if not isinstance(test_syn_comp_field, str):

                        print("TypeError in delay parameters: the value of the key 'synComp' must be of type 'str'. The current type is '%s'." % (type(test_syn_comp_field)))

                        error_counter += 1

                    else:

                        if test_syn_comp_field != 'all':

                            try:

                                test_syn_ends_with_field = delay_params[delay_param]['synEndsWith']

                                if not isinstance(test_syn_ends_with_field, list):

                                    print("TypeError in delay parameters: the value of the key 'synEndsWith' must be of type 'list'.")
                                    print("The current type is '%s'." % (type(test_syn_ends_with_field)))

                                    error_counter += 1

                            except KeyError:

                                print("KeyError in delay parameters: the key 'synEndsWith' is not in the keys of delay parameter dictionary.")

                                error_counter += 1

                            try:

                                test_target_cell_group = delay_params[delay_param]['targetCellGroup']

                                if not isinstance(test_target_cell_group, list):

                                    print("TypeError in delay parameters: the value of the key 'targetCellGroup' must be of type 'list'.")
                                    print("The current type is '%s'." % (type(test_target_cell_group)))

                                    error_counter += 1

                            except KeyError:

                                print("KeyError in delay parameters: the key 'targetCellGroup' is not in the keys of delay parameter dictionary.")

                                error_counter += 1

                except KeyError:

                    print("KeyError in delay parameters: the key 'synComp' is not in the keys of delay parameter dictionary.")

                    error_counter += 1

    if error_counter == 0:

        return True

    else:

        return False              


##############################################################################################

def check_inputs(input_params, popDict, path_to_cells, path_to_synapses=None):

    error_counter = 0

    for cell_receiver in input_params.keys():

        found_target_pop = False

        for pop_id in popDict.keys():

            if cell_receiver in pop_id:

                test_cell_component = popDict[pop_id]

                if test_cell_component['Compartments'] == 'single':

                    cell_type = None

                if test_cell_component['Compartments'] == 'multi':

                    segment_groups = get_segment_groups(test_cell_component['PopObj'].component, path_to_cells)

                    cell_type = test_cell_component['PopObj'].component

                found_target_pop = True

                break

        if not found_target_pop:

            opencortex.print_comment_v("KeyError in input parameters: cell population id '%s' is not in the keys of population dictionary" % cell_receiver)

            error_counter += 1

            cell_type = None

        if not isinstance(input_params[cell_receiver], list):

            opencortex.print_comment_v("TypeError in input parameters: the dictionary value for '%s' must be a list."
                                       " The current type is %s" % (cell_receiver, type(input_params[cell_receiver])))

            error_counter += 1

        else:

            for input_group_ind in range(0, len(input_params[cell_receiver])):

                input_group_params = input_params[cell_receiver][input_group_ind]

                try:

                    test_key = input_group_params['InputType']

                    if not isinstance(test_key, str):

                        opencortex.print_comment_v("TypeError in input parameters: the value of the key 'InputType' must be of type 'string'. The current type is %s" % type(test_key))

                        error_counter += 1 

                    if test_key not in ['GeneratePoissonTrains', 'PulseGenerators', 'GenerateSpikeSourcePoisson']:

                        opencortex.print_comment_v("ValueError in input parameters: the value of the key 'InputType' must be one of the following: "   
                                                   "'GeneratePoissonTrains','PulseGenerators', 'GenerateSpikeSourcePoisson'")

                        error_counter += 1

                    else:

                        if test_key == "GeneratePoissonTrains":

                            try:
                                test_train_type = input_group_params['TrainType']

                                if not isinstance(test_train_type, str):

                                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'TrainType' must be of type 'string'. "
                                                               "The current type is %s" % type(test_train_type))

                                    error_counter += 1
                                else:

                                    if test_train_type not in ['persistent', 'transient']:

                                        opencortex.print_comment_v("ValueError in input parameters: the value of the key 'TrainType' when 'InputType' is 'GeneratePoissonTrains' must be"
                                                                   " one of the following: 'persistent' or 'transient'")

                                        error_counter += 1

                                    else:

                                        if test_train_type == "persistent":

                                            try:

                                                test_rates = input_group_params['AverageRateList']

                                                if not isinstance(test_rates, list):

                                                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'AverageRateList' must be of type 'list'."
                                                                               " The current type is %s" % type(test_rates))

                                                    error_counter += 1

                                                else:
                                                    for r in range(0, len(test_rates)):

                                                        if not isinstance(test_rates[r], float):

                                                            opencortex.print_comment_v("TypeError in input parameters: the list values of the key 'AverageRateList' must be of type 'float'."
                                                                                       " The current type is %s" % type(test_rates[r]))

                                                            error_counter += 1

                                            except KeyError:

                                                opencortex.print_comment_v("KeyError in input parameters: the key 'AverageRateList' is not in the keys of input parameters.")

                                                error_counter += 1

                                        if test_train_type == "transient":

                                            try:

                                                test_time_units = input_group_params['TimeUnits']

                                                if not isinstance(test_time_units, str):

                                                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'TimeUnits' must be of type 'str'."
                                                                               " The current type is %s." % type(test_time_units)) 

                                                    error_counter += 1

                                            except KeyError:

                                                opencortex.print_comment_v("KeyError in input parameters: the key 'TimeUnits' is not in the keys of input parameters.")

                                                error_counter += 1


                                            try:

                                                test_rates = input_group_params['AverageRateList']

                                                if not isinstance(test_rates, list):

                                                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'AverageRateList' must be of type 'list'."
                                                                               " The current type is %s." % type(test_rates))

                                                    error_counter += 1

                                                else:

                                                    for r in range(0, len(test_rates)):

                                                        if not isinstance(test_rates[r], float):

                                                            opencortex.print_comment_v("TypeError in input parameters: the list values of the key 'AverageRateList' must be of type 'float'."
                                                                                       " The current type is %s." % type(test_rates[r]))

                                                            error_counter += 1

                                            except KeyError:

                                                opencortex.print_comment_v("KeyError in input parameters: the key 'AverageRateList' is not in the keys of input parameters.")

                                                error_counter += 1

                                            try:

                                                test_rates = input_group_params['DelayList']

                                                if not isinstance(test_rates, list):

                                                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'DelayList' must be of type 'list'."
                                                                               " The current type is %s." % type(test_rates))

                                                    error_counter += 1

                                                else:

                                                    for r in range(0, len(test_rates)):

                                                        if not isinstance(test_rates[r], float):

                                                            opencortex.print_comment_v("TypeError in input parameters: the list values of the key 'DelayList' must be of type 'float'."
                                                                                       " The current type is %s." % type(test_rates[r]))

                                                            error_counter += 1

                                            except KeyError:

                                                opencortex.print_comment_v("KeyError in input parameters: the key 'DelayList' is not in the keys of input parameters.")

                                                error_counter += 1

                                            try:

                                                test_rates = input_group_params['DurationList']

                                                if not isinstance(test_rates, list):

                                                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'DurationList' must be of type 'list'."
                                                                               " The current type is %s." % type(test_rates))

                                                    error_counter += 1

                                                else:

                                                    for r in range(0, len(test_rates)):

                                                        if not isinstance(test_rates[r], float):

                                                            opencortex.print_comment_v("TypeError in input parameters: the list values of the key 'DurationList' must be of type 'float'."
                                                                                       " The current type is %s." % type(test_rates[r]))

                                                            error_counter += 1

                                            except KeyError:

                                                opencortex.print_comment_v("KeyError in input parameters: the key 'DurationList' is not in the keys of input parameters.")

                                                error_counter += 1

                            except KeyError:

                                opencortex.print_comment_v("KeyError in input parameters: the key 'TrainType' is not in the keys of input parameters.")

                                error_counter += 1

                            try:

                                test_rate_units = input_group_params['RateUnits']

                                if not isinstance(test_rate_units, str):

                                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'RateUnits' must be of type 'str'."
                                                               "The current type is %s." % type(test_rate_units))

                                    error_counter += 1

                            except KeyError:

                                opencortex.print_comment_v("KeyError in input parametres: the key 'RateUnits' is not in the keys of inputs parameters.")

                                error_counter += 1

                            try:

                                test_synapse = input_group_params['Synapse']

                                if not isinstance(test_synapse, str):

                                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'Synapse' must be of type 'str'."
                                                               " The current type is %s." % type(test_synapse))

                                    error_counter += 1

                                else:

                                    if path_to_synapses != None:

                                        found = check_synapse_location(test_synapse, path_to_synapses)

                                        if not found:

                                            opencortex.print_comment_v("ValueError in input parameters: the value '%s' of the key 'Synapse' is not found in %s" % (test_synapse, path_to_synapses))

                                            error_counter += 1

                            except KeyError:

                                opencortex.print_comment_v("KeyError in input parameters: the key 'Synapse' is not in the keys of input parameters.")  

                                error_counter += 1

                        if test_key == 'GenerateSpikeSourcePoisson':

                            try:

                                test_time_units = input_group_params['TimeUnits']

                                if not isinstance(test_time_units, str):

                                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'TimeUnits' must be of type 'str'."
                                                               " The current type is %s." % type(test_time_units)) 

                                    error_counter += 1

                            except KeyError:

                                opencortex.print_comment_v("KeyError in input parameters: the key 'TimeUnits' is not in the keys of input parameters.")

                                error_counter += 1

                            try:

                                test_rates = input_group_params['AverageRateList']

                                if not isinstance(test_rates, list):

                                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'AverageRateList' must be of type 'list'."
                                                               " The current type is %s." % type(test_rates))

                                    error_counter += 1

                                else:

                                    for r in range(0, len(test_rates)):

                                        if not isinstance(test_rates[r], float):

                                            opencortex.print_comment_v("TypeError in input parameters: the list values of the key 'AverageRateList' must be of type 'float'."
                                                                       " The current type is %s." % type(test_rates[r]))

                                            error_counter += 1

                            except KeyError:

                                opencortex.print_comment_v("KeyError in input parameters: the key 'AverageRateList' is not in the keys of input parameters.")

                                error_counter += 1

                            try:

                                test_rates = input_group_params['DelayList']

                                if not isinstance(test_rates, list):

                                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'DelayList' must be of type 'list'."
                                                               " The current type is %s." % type(test_rates))

                                    error_counter += 1

                                else:

                                    for r in range(0, len(test_rates)):

                                        if not isinstance(test_rates[r], float):

                                            opencortex.print_comment_v("TypeError in input parameters: the list values of the key 'DelayList' must be of type 'float'."
                                                                       " The current type is %s." % type(test_rates[r]))

                                            error_counter += 1

                            except KeyError:

                                opencortex.print_comment_v("KeyError in input parameters: the key 'DelayList' is not in the keys of input parameters.")

                                error_counter += 1

                            try:

                                test_rates = input_group_params['DurationList']

                                if not isinstance(test_rates, list):

                                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'DurationList' must be of type 'list'."
                                                               " The current type is %s." % type(test_rates))

                                    error_counter += 1

                                else:

                                    for r in range(0, len(test_rates)):

                                        if not isinstance(test_rates[r], float):

                                            opencortex.print_comment_v("TypeError in input parameters: the list values of the key 'DurationList' must be of type 'float'."
                                                                       " The current type is %s." % type(test_rates[r]))

                                            error_counter += 1

                            except KeyError:

                                opencortex.print_comment_v("KeyError in input parameters: the key 'DurationList' is not in the keys of input parameters.")

                                error_counter += 1

                            try:

                                test_weights = input_group_params['WeightList']

                                if not isinstance(test_weights, list):

                                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'WeightList' must be of type 'list'."
                                                               " The current type is %s." % type(test_weights))

                                    error_counter += 1

                                else:

                                    for r in range(0, len(test_weights)):

                                        if not isinstance(test_weights[r], float):

                                            opencortex.print_comment_v("TypeError in input parameters: the list values of the key 'WeightList' must be of type 'float'."
                                                                       " The current type is %s." % type(test_weights[r]))

                                            error_counter += 1

                            except KeyError:

                                opencortex.print_comment_v("KeyError in input parameters: the key 'WeightList' is not in the keys of input parameters.")

                                error_counter += 1

                            try:

                                test_rate_units = input_group_params['RateUnits']

                                if not isinstance(test_rate_units, str):

                                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'RateUnits' must be of type 'str'."
                                                               "The current type is %s." % type(test_rate_units))

                                    error_counter += 1

                            except KeyError:

                                opencortex.print_comment_v("KeyError in input parametres: the key 'RateUnits' is not in the keys of inputs parameters.")

                                error_counter += 1

                            try:

                                test_synapse = input_group_params['Synapse']

                                if not isinstance(test_synapse, str):

                                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'Synapse' must be of type 'str'."
                                                               " The current type is %s." % type(test_synapse))

                                    error_counter += 1

                                else:

                                    if path_to_synapses != None:

                                        found = check_synapse_location(test_synapse, path_to_synapses)

                                        if not found:

                                            opencortex.print_comment_v("ValueError in input parameters: the value '%s' of the key 'Synapse' is not found in %s" % (test_synapse, path_to_synapses))

                                            error_counter += 1

                            except KeyError:

                                opencortex.print_comment_v("KeyError in input parameters: the key 'Synapse' is not in the keys of input parameters.")  

                                error_counter += 1

                        if test_key == 'PulseGenerators':

                            try:

                                test_time_units = input_group_params['TimeUnits']

                                if not isinstance(test_time_units, str):

                                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'TimeUnits' must be of type 'str'."
                                                               " The current type is %s." % type(test_time_units)) 

                                    error_counter += 1

                            except KeyError:

                                opencortex.print_comment_v("KeyError in input parameters: the key 'TimeUnits' is not in the keys of input parameters.")

                                error_counter += 1

                            try:

                                test_amplitude_units = input_group_params['AmplitudeUnits']

                                if not isinstance(test_amplitude_units, str):

                                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'AmplitudeUnits' must be of type 'str'."
                                                               " The current type is %s." % type(test_amplitude_units))

                                    error_counter += 1

                            except KeyError:

                                opencortex.print_comment_v("KeyError in input parametres: the key 'AmplitudeUnits' is not in the keys of inputs parameters.")

                                error_counter += 1

                            try:

                                test_noise = input_group_params['Noise']

                                if not isinstance(test_noise, bool):

                                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'Noise' must be of type 'bool'."
                                                               " The current type is %s." % type(test_noise))

                                    error_counter += 1

                                else:

                                    if test_noise:

                                        try:

                                            test_smallest_amplitudes = input_group_params['SmallestAmplitudeList']

                                            if not isinstance(test_smallest_amplitudes, list):

                                                opencortex.print_comment_v("TypeError in input parameters: the value of the key 'SmallestAmplitudeList' must be of type 'list'."
                                                                           " The current type is %s." % type(test_smallest_amplitudes))

                                                error_counter += 1

                                        except KeyError:

                                            opencortex.print_comment_v("KeyError in input parameters: the key 'SmallestAmplitudeList' is not in the keys of input parameters when "
                                                                       "'Noise' is set to True.")

                                            error_counter += 1    

                                        try:

                                            test_largest_amplitudes = input_group_params['LargestAmplitudeList']

                                            if not isinstance(test_largest_amplitudes, list):

                                                opencortex.print_comment_v("TypeError in input parameters: the value of the key 'LargestAmplitudeList' must be of type 'list'."
                                                                           " The current type is %s." % type(test_largest_amplitudes))

                                                error_counter += 1

                                        except KeyError:

                                            opencortex.print_comment_v("KeyError in input parameters: the key 'LargestAmplitudeList' is not in the keys of input parameters when "
                                                                       "'Noise' is set to True.")

                                            error_counter += 1

                                    else:

                                        try:

                                            test_amplitudes = input_group_params['AmplitudeList']

                                            if not isinstance(test_amplitudes, list):

                                                opencortex.print_comment_v("TypeError in input parameters: the value of the key 'AmplitudeList' must be of type 'list'."
                                                                           " The current type is %s." % type(test_amplitudes))

                                                error_counter += 1

                                            else:

                                                for r in range(0, len(test_amplitudes)):

                                                    if not isinstance(test_amplitudes[r], float):

                                                        opencortex.print_comment_v("TypeError in input parameters: the list values of the key 'AverageRateList' must be of type 'float'."
                                                                                   " The current type is %s." % type(test_amplitudes[r]))

                                                        error_counter += 1

                                        except KeyError:

                                            opencortex.print_comment_v("KeyError in input parameters: the key 'AmplitudeList' is not in the keys of input parameters.")

                                            error_counter += 1

                            except KeyError:

                                opencortex.print_comment_v("KeyError in input parameters: the key 'Noise' is not in the keys of input parameters.")

                                error_counter += 1

                            try:
                                test_delays = input_group_params['DelayList']

                                if not isinstance(test_delays, list):

                                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'DelayList' must be of type 'list'."
                                                               " The current type is %s." % type(test_delays))

                                    error_counter += 1

                                else:

                                    for r in range(0, len(test_delays)):

                                        if not isinstance(test_delays[r], float):

                                            opencortex.print_comment_v("TypeError in input parameters: the list values of the key 'DelayList' must be of type 'float'."
                                                                       " The current type is %s." % type(test_delays[r]))

                                            error_counter += 1

                            except KeyError:

                                opencortex.print_comment_v("KeyError in input parameters: the key 'DelayList' is not in the keys of input parameters.")

                                error_counter += 1

                            try:

                                test_durations = input_group_params['DurationList']

                                if not isinstance(test_durations, list):

                                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'DurationList' must be of type 'list'."
                                                               " The current type is %s." % type(test_durations))

                                    error_counter += 1

                                else:

                                    for r in range(0, len(test_durations)):

                                        if not isinstance(test_durations[r], float):

                                            opencortex.print_comment_v("TypeError in input parameters: the list values of the key 'DurationList' must be of type 'float'."
                                                                       " The current type is %s." % type(test_durations[r]))

                                            error_counter += 1

                            except KeyError:

                                opencortex.print_comment_v("KeyError in input parameters: the key 'DurationList' is not in the keys of input parameters.")

                                error_counter += 1


                except KeyError:

                    opencortex.print_comment_v("KeyError in input parameters: the key 'InputType' is not in input parameters.")

                    error_counter += 1

                try:

                    test_input_name = input_group_params['InputName']

                    if not isinstance(test_input_name, str):

                        opencortex.print_comment_v("TypeError in input parameters: the value of the key 'InputName' must be of type 'str'."

                                                   "The current type is %s." % type(test_input_name))

                        error_counter += 1

                except KeyError:

                    opencortex.print_comment_v("KeyError in input parameters: the key 'InputName' is not in input parameters.")   

                    error_counter += 1

                if ('UniversalTargetSegmentID' not in input_group_params.keys()) and ('UniversalFractionAlong' not in input_group_params.keys()):

                    try:

                        test_key = input_group_params['TargetDict']

                        if not isinstance(test_key, dict):

                            opencortex.print_comment_v("TypeError in input parameters: the value of the key 'TargetDict' in input parameters must be of type 'dict'."
                                                       " The current type is %s." % type(test_key)) 

                            error_counter += 1

                        else:

                            if cell_type != None:

                                for target_segment_group in test_key.keys():

                                    if not check_segment_group(segment_groups, target_segment_group):

                                        opencortex.print_comment_v("ValueError in input parameters: '%s' is not a segment group of the cell type '%s'" % (target_segment_group, cell_receiver))

                                        error_counter += 1

                                    else:

                                        if not isinstance(test_key[target_segment_group], int):

                                            opencortex.print_comment_v("TypeError in input parameters: the value of the key '%s' must be of type 'int'."
                                                                       " The current type is %s" % (target_segment_group, type(test_key[target_segment_group])))

                                            error_counter += 1

                    except KeyError:

                        opencortex.print_comment_v("KeyError in input parameters: the key 'TargetDict' is not in input parameters.")

                        error_counter += 1

                if 'TargetDict' in input_group_params.keys():

                    if 'UniversalTargetSegmentID' in input_group_params.keys():

                        opencortex.print_comment_v("KeyError in input parameters: the key 'UniversalTargetSegmentID' cannot be specified together with the key 'TargetDict'.")

                        error_counter += 1

                    if 'UniversalFractionAlong' in input_group_params.keys():

                        opencortex.print_comment_v("KeyError in input parameters: the key 'UniversalFractionAlong' cannot be specified together with the key 'TargetDict'.")

                        error_counter += 1

                else:

                    try:

                        test_target_seg_id = input_group_params['UniversalTargetSegmentID']

                    except KeyError:

                        opencortex.print_comment_v("KeyError in input parameters: the key 'UniversalTargetSegmentID' must be specified when the key 'TargetDict' is not in "
                                                   "input parameters.")

                        error_counter += 1

                    try:

                        test_fraction_along = input_group_params['UniversalFractionAlong']

                    except KeyError:

                        opencortex.print_comment_v("KeyError in input parameters: the key 'UniversalFractionAlong' must be specified when the key 'TargetDict' is not in "
                                                   "input parameters.")

                        error_counter += 1

                try:

                    test_key = input_group_params['FractionToTarget']

                    if not isinstance(test_key, float):

                        opencortex.print_comment_v("TypeError: the value of the key 'FractionToTarget' must be of type 'float'. The current type is %s." % type(test_key))

                        error_counter += 1

                except KeyError:

                    opencortex.print_comment_v("KeyError: the key 'FractionToTarget' is not in input parameters.")

                    error_counter += 1


                try:

                    test_key = input_group_params['LocationSpecific']


                    if not isinstance(test_key, bool):

                        opencortex.print_comment_v("TypeError in input parameters: the value of the key 'LocationSpecific' must be of the type 'bool'."
                                                   " The current type is %s." % type(test_key)) 

                        error_counter += 1

                    else:

                        if test_key:

                            try:

                                test_region_key = input_group_params['TargetRegions']

                                if not isinstance(test_region_key, list):

                                    opencortex.print_comment_v("TypeError in input parameters: the value of the key 'TargetRegions' must be of the type 'list'."
                                                               " The current type is %s." % type(test_region_key))

                                    error_counter += 1

                                else:

                                    for region in range(0, len(test_region_key)):

                                        if not isinstance(test_region_key[region], dict):

                                            opencortex.print_comment_v("TypeError in input parameters: the list values of the key 'TargetRegions' must be of the type 'dict'."
                                                                       " The current type is %s." % type(test_region_key[region]))

                                            error_counter += 1

                                        else:

                                            for dim_key in ['XVector', 'YVector', 'ZVector']:

                                                if dim_key not in test_region_key[region].keys():

                                                    opencortex.print_comment_v("ValueError in input parameters: the list values of the key 'TargetRegions' must be dictionaries "
                                                                               "with the following keys: 'XVector', 'YVector', 'ZVector'.")

                                                    error_counter += 1 

                                                else:

                                                    if not isinstance(test_region_key[region][dim_key], list):

                                                        opencortex.print_comment_v("TypeError in input parameters: the 'X/Y/ZVector' must store the value of type 'list'."
                                                                                   " The current type is %s." % type(test_region_key[region][dim_key]))

                                                        error_counter += 1

                                                    else:

                                                        if len(test_region_key[region][dim_key]) != 2:

                                                            opencortex.print_comment_v("ValueError in input parameters: the lists stored by 'XVector', 'YVector' and 'ZVector'"
                                                                                       " must contain two values.")

                                                            error_counter += 1

                                                        else:

                                                          if (test_region_key[region][dim_key][0]-test_region_key[region][dim_key][1]) == 0:

                                                             opencortex.print_comment_v("ValueError in input parameters: the lists stored by 'XVector', 'YVector' and 'ZVector'"
                                                             " must contain two different values.")

                                                             error_counter += 1

                            except KeyError:

                              opencortex.print_comment_v("KeyError in input parameters: 'LocationSpecific' is True but the key 'TargetRegions' is not in input parameters.")

                              error_counter += 1



                except KeyError:

                   opencortex.print_comment_v("KeyError in input parameters: the key 'LocationSpecific' is not in input parameters.")

                   error_counter += 1

    if error_counter == 0:

       return True

    else:

       return False       

##############################################################################################          

