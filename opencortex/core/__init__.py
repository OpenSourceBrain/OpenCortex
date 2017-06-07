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


def include_cell_prototype(nml_doc, cell_nml2_path):
    
    """
    Add a NeuroML2 file containing a cell definition
    """

    nml_doc.includes.append(neuroml.IncludeType(cell_nml2_path)) 


##############################################################################################


def include_neuroml2_file(nml_doc, nml2_file_path):

    """
    Add a NeuroML2 file containing definitions of elements which can be used in the network
    """
    nml_doc.includes.append(neuroml.IncludeType(nml2_file_path)) 
    
    
#########################################################################################

def include_neuroml2_cell(nml_doc, cell_nml2_path, cell_id, channels_also=True):
    
    """
    Add a cell with id `cell_id` which is in `cell_nml2_path` to the build document,
    along with all its channels (if channels_also==True)
    """

    return oc_build._include_neuroml2_cell(nml_doc, cell_nml2_path, cell_id, channels_also=channels_also)
    
    
#########################################################################################

def include_neuroml2_cell_and_channels(nml_doc, cell_nml2_path, cell_id):
    
    """
    TODO: remove, due to include_neuroml2_cell
    Add a cell with id `cell_id` which is in `cell_nml2_path` to the build document,
    along with all its channels
    """

    return oc_build._include_neuroml2_cell(nml_doc, cell_nml2_path, cell_id, channels_also=True)


#########################################################################################

def include_opencortex_cell(nml_doc, reference):
    
    """
    Include a cell from the standard set of NeuroML2 cells included with OpenCortex.
    See https://github.com/OpenSourceBrain/OpenCortex/tree/master/NeuroML2/prototypes.
    """

    cell_id = reference.split('/')[1].split('.')[0]
    return oc_build._add_cell_and_channels(nml_doc, reference, cell_id, use_prototypes=True)



##############################################################################################

def add_exp_two_syn(nml_doc, id, gbase, erev, tau_rise, tau_decay):
    
    """
    Adds an <expTwoSynapse> element to the document. See the definition of the 
    behaviour of this here: https://www.neuroml.org/NeuroML2CoreTypes/Synapses.html#expTwoSynapse
    
    Returns the class created.
    """
    syn0 = neuroml.ExpTwoSynapse(id=id, gbase=gbase,
                                 erev=erev,
                                 tau_rise=tau_rise,
                                 tau_decay=tau_decay)

    nml_doc.exp_two_synapses.append(syn0)

    return syn0


##############################################################################################

def add_gap_junction_synapse(nml_doc, id, conductance):
    
    """
    Adds a <gapJunction> element to the document. See the definition of the 
    behaviour of this here: https://www.neuroml.org/NeuroML2CoreTypes/Synapses.html#gapJunction
    
    Returns the class created.
    """
    
    syn0 = neuroml.GapJunction(id=id, conductance=conductance)

    nml_doc.gap_junctions.append(syn0)

    return syn0


##############################################################################################

def add_poisson_firing_synapse(nml_doc, id, average_rate, synapse_id):
    
    """
    Adds a <poissonFiringSynapse> element to the document. See the definition of the 
    behaviour of this here: https://www.neuroml.org/NeuroML2CoreTypes/Inputs.html#poissonFiringSynapse
    
    Returns the class created.
    """

    return oc_build._add_poisson_firing_synapse(nml_doc, id, average_rate, synapse_id)


#########################################################################

def add_transient_poisson_firing_synapse(nml_doc, id, average_rate, delay, duration, synapse_id):

    """
    Adds a <transientPoissonFiringSynapse> element to the document. See the definition of the 
    behaviour of this here: https://www.neuroml.org/NeuroML2CoreTypes/Inputs.html#transientPoissonFiringSynapse
    
    Returns the class created.
    """
    
    return oc_build._add_transient_poisson_firing_synapse(nml_doc, id, average_rate, delay, duration, synapse_id)


################################################################################    

def add_pulse_generator(nml_doc, id, delay, duration, amplitude):

    """
    Adds a <pulseGenerator> element to the document. See the definition of the 
    behaviour of this here: https://www.neuroml.org/NeuroML2CoreTypes/Inputs.html#pulseGenerator
    
    Returns the class created.
    """

    return oc_build._add_pulse_generator(nml_doc, id, delay, duration, amplitude)


##############################################################################################

def add_spike_source_poisson(nml_doc, id, start, duration, rate):

    """
    Adds a <SpikeSourcePoisson> element to the document. See the definition of the 
    behaviour of this here: https://www.neuroml.org/NeuroML2CoreTypes/PyNN.html#SpikeSourcePoisson
    
    Returns the class created.
    """
    
    return oc_build._add_spike_source_poisson(nml_doc, id, start, duration, rate)



##############################################################################################

def add_single_cell_population(net, pop_id, cell_id, x=0, y=0, z=0, color=None):
    
    """
    Add a population with id `pop_id` containing a single instance of cell `cell_id`. 
    Optionally specify (`x`,`y`,`z`) and the population `color`.
    """

    pop = neuroml.Population(id=pop_id, component=cell_id, type="populationList", size=1)
    if color is not None:
        pop.properties.append(Property("color", color))
    net.populations.append(pop)

    inst = neuroml.Instance(id=0)
    pop.instances.append(inst)
    inst.location = neuroml.Location(x=x, y=y, z=z)

    return pop


##############################################################################################

def add_population_in_rectangular_region(net, 
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
    Method which creates a cell population in the NeuroML2 network and distributes these cells in the rectangular region. Input arguments are:

    `net`
        reference to the network object previously created

    `pop_id` 
        population id

    `cell_id`
        cell component id

    `size` 
        size of a population

    `x_min` 
        lower x bound of a rectangular region

    `y_min` 
        lower y bound of a rectangular region

    `z_min` 
        lower z bound of a rectangular region

    `x_size` 
        width of a rectangular region along x axis

    `y_size` 
        width of a rectangular region along y axis

    `z_size` 
        width of a rectangular region along z axis

    `cell_bodies_overlap` 
        boolean value which defines whether cell somata can overlap; default is set to True

    `store_soma` 
        boolean value which specifies whether soma positions have to be returned in the output array; default is set to False

    `population_dictionary` 
        optional argument in the format returned by opencortex.utils.add_populations_in_rectangular_layers; default value is None but it must be specified when cell_bodies_overlap is set to False

    `cell_diameter_dict` 
        optional argument in the format {'cell_id1': soma diameter of type 'float', 'cell_id2': soma diameter of type 'float'}; default is None but it must be specified when cell_bodies_overlap is set to False

    `color` 
        optional color, default is None

    """

    return oc_build._add_population_in_rectangular_region(net, 
                                         pop_id, 
                                         cell_id, 
                                         size, 
                                         x_min, 
                                         y_min, 
                                         z_min, 
                                         x_size, 
                                         y_size, 
                                         z_size,
                                         cell_bodies_overlap,
                                         store_soma,
                                         population_dictionary,
                                         cell_diameter_dict,
                                         color)
    
    
    

##############################################################################################     

def add_probabilistic_projection(net, 
                                 prefix, 
                                 presynaptic_population, 
                                 postsynaptic_population, 
                                 synapse_id, 
                                 connection_probability,
                                 delay=0,
                                 weight=1):     
    """
    Add a projection between `presynaptic_population` and `postsynaptic_population` with probability of connection between each pre & post pair of cells given by `connection_probability`. Attributes:
    
    `net`
        reference to the network object previously created
        
    `prefix`
        prefix to use in the id of the projection
    
    `presynaptic_population`
        presynaptic population e.g. added via add_population_in_rectangular_region()
    
    `postsynaptic_population`
        postsynaptic population e.g. added via add_population_in_rectangular_region()
    
    `synapse_id`
        id of synapse previously added, e.g. added with add_exp_two_syn()
        
    `connection_probability`
        For each pre syn cell i and post syn cell j, where i!=j, the chance they will be connected is given by this
        
    `delay`
        optional delay for each connection, default 0 ms
        
    `weight`
        optional weight for each connection, default 1
    
    """

    if presynaptic_population.size == 0 or postsynaptic_population.size == 0:
        return None

    proj = neuroml.Projection(id="%s_%s_%s" % (prefix, presynaptic_population.id, postsynaptic_population.id), 
                              presynaptic_population=presynaptic_population.id, 
                              postsynaptic_population=postsynaptic_population.id, 
                              synapse=synapse_id)


    count = 0

    for i in range(0, presynaptic_population.size):
        for j in range(0, postsynaptic_population.size):
            if i != j or presynaptic_population.id != postsynaptic_population.id:
                if connection_probability >= 1 or random.random() < connection_probability:
                    oc_build._add_connection(proj, 
                                   count, 
                                 presynaptic_population, 
                                   i, 
                                   0, 
                                 postsynaptic_population, 
                                   j, 
                                   0,
                                   delay=delay,
                                   weight=weight)
                    count += 1

    net.projections.append(proj)

    return proj


##############################################################################################

def add_targeted_projection(nml_doc,
                         net,
                         prefix,
                         presynaptic_population,
                         postsynaptic_population,
                         targeting_mode,
                         synapse_list,
                         number_conns_per_cell,
                         pre_segment_group,
                         post_segment_group,
                         delays_dict=None,
                         weights_dict=None):
    '''
    Adds (chemical, event based) projection from `presynaptic_population` to `postsynaptic_population`, 
    specifically limiting connections presynaptically to `pre_segment_group` and postsynaptically to `post_segment_group`.
    '''

    if presynaptic_population.size == 0 or postsynaptic_population.size == 0:
        return None

    projections = []


    pre_cell = oc_build.cell_ids_vs_nml_docs[presynaptic_population.component].get_by_id(presynaptic_population.component)
    post_cell = oc_build.cell_ids_vs_nml_docs[postsynaptic_population.component].get_by_id(postsynaptic_population.component)


    pre_segs = oc_build.extract_seg_ids(pre_cell,
                               [pre_segment_group],
                               "segGroups")
    post_segs = oc_build.extract_seg_ids(post_cell,
                                [post_segment_group],
                                "segGroups")


    pre_seg_target_dict = oc_build.make_target_dict(pre_cell, pre_segs)
    post_seg_target_dict = oc_build.make_target_dict(post_cell, post_segs)
    #print pre_seg_target_dict, post_seg_target_dict

    for synapse in synapse_list:

        proj_id = "%s_%s_%s" % (prefix, presynaptic_population.id, postsynaptic_population.id) if len(synapse_list) == 1 else \
            "%s_%s_%s_%s" % (prefix, presynaptic_population.id, postsynaptic_population.id, synapse)

        opencortex.print_comment_v("Adding projection: %s: %s (%s) -> %s (%s)" % (proj_id, pre_cell.id, pre_segs, post_cell.id, post_segs))

        proj = neuroml.Projection(id=proj_id, 
                                  presynaptic_population=presynaptic_population.id, 
                                  postsynaptic_population=postsynaptic_population.id, 
                                  synapse=synapse)


        projections.append(proj)

    subset_dict = {}#{'dendrite_group':number_conns_per_cell}

    subset_dict[post_segment_group] = number_conns_per_cell


    oc_build.add_targeted_projection_by_dicts(net,
                        projections,
                        presynaptic_population,
                        postsynaptic_population,
                        targeting_mode,
                        synapse_list,
                        pre_seg_target_dict,
                        post_seg_target_dict,
                        subset_dict,
                        delays_dict,
                        weights_dict)

    return projections 


############################################################################################## 

def add_targeted_electrical_projection(nml_doc,
                          net,
                          prefix,
                          presynaptic_population,
                          postsynaptic_population,
                          targeting_mode,
                          synapse_list,
                          number_conns_per_cell,
                          pre_segment_group,
                          post_segment_group):

    '''
    Adds (electrical, gap junction mediated) projection from `presynaptic_population` to `postsynaptic_population`, 
    specifically limiting connections presynaptically to `pre_segment_group` and postsynaptically to `post_segment_group`.
    '''

    if presynaptic_population.size == 0 or postsynaptic_population.size == 0:
        return None

    projections = []

    pre_cell = oc_build.cell_ids_vs_nml_docs[presynaptic_population.component].get_by_id(presynaptic_population.component)
    post_cell = oc_build.cell_ids_vs_nml_docs[postsynaptic_population.component].get_by_id(postsynaptic_population.component)


    pre_segs = oc_build.extract_seg_ids(pre_cell,
                               [pre_segment_group],
                               "segGroups")
    post_segs = oc_build.extract_seg_ids(post_cell,
                                [post_segment_group],
                                "segGroups")


    pre_seg_target_dict = oc_build.make_target_dict(pre_cell, pre_segs)
    post_seg_target_dict = oc_build.make_target_dict(post_cell, post_segs)
    #print pre_seg_target_dict, post_seg_target_dict

    for synapse in synapse_list:

        proj_id = "%s_%s_%s" % (prefix, presynaptic_population.id, postsynaptic_population.id) if len(synapse_list) == 1 else \
            "%s_%s_%s_%s" % (prefix, presynaptic_population.id, postsynaptic_population.id, synapse)

        opencortex.print_comment_v("Adding projection: %s: %s (%s) -> %s (%s)" % (proj_id, pre_cell.id, pre_segs, post_cell.id, post_segs))

        proj = neuroml.ElectricalProjection(id=proj_id, 
                                            presynaptic_population=presynaptic_population.id, 
                                            postsynaptic_population=postsynaptic_population.id)


        projections.append(proj)

    subset_dict = {}#{'dendrite_group':number_conns_per_cell}

    subset_dict[post_segment_group] = number_conns_per_cell


    oc_build._add_elect_projection(net,
                         projections,
                         presynaptic_population,
                         postsynaptic_population,
                         targeting_mode,
                         synapse_list,
                         pre_seg_target_dict,
                         post_seg_target_dict,
                         subset_dict)

    return projections 



##############################################################################################

def add_inputs_to_population(net, 
                             id, 
                             population, 
                             input_comp_id, 
                             number_per_cell=1, 
                             all_cells=False, 
                             only_cells=None,
                             segment_ids=[0],
                             fraction_alongs=[0.5]):
    
    """
    Add current input to the specified population. Attributes:
    
    `net`
        reference to the network object previously created

    `id` 
        id of the <inputList> to be created
        
    `population` 
        the <population> to be targeted
        
    `input_comp_id` 
        id of the component to be used for the input (e.g. added with add_pulse_generator())
        
    `number_per_cell`
        how many inputs to apply to each cell of the population. Default 1
        
    `all_cells`
        Whether to target all cells. Default False
        
    `only_cells`
        Which specific cells to target. List of ids. Default None
        
    `segment_ids`
        List of segment ids to place inputs onto on each cell. Either list of 1 value or list of number_per_cell entries. Default [0]
        
    `fraction_alongs`
        List of fractions along the specified segments to place inputs onto on each cell. Either list of 1 value or list of number_per_cell entries. Default [0.5]
        
        
    """

    if all_cells and only_cells is not None:
        error = "Error! Method opencortex.build.%s() called with both arguments all_cells and only_cells set!" % sys._getframe().f_code.co_name
        opencortex.print_comment_v(error)
        raise Exception(error)
    
    if len(segment_ids)!=1 or len(segment_ids)!=number_per_cell:
        
        error = "Error! Attribute segment_ids in method opencortex.build.%s()"% sys._getframe().f_code.co_name+\
        " should be a list of one integer (id of the segment all inputs to each cell go into) or a "+ \
        "list of the same length as number_per_cell!" 
        opencortex.print_comment_v(error)
        raise Exception(error)
    
    if len(fraction_alongs)!=1 or len(fraction_alongs)!=number_per_cell:
        
        error = "Error! Attribute fraction_alongs in method opencortex.build.%s()"% sys._getframe().f_code.co_name+\
        " should be a list of one float (fraction along the segment all inputs to each cell go into) or a "+ \
        "list of the same length as number_per_cell!" 
        opencortex.print_comment_v(error)
        raise Exception(error)

    cell_ids = []

    if all_cells:
        cell_ids = range(population.size)
    if only_cells is not None:
        if only_cells == []:
            return
        cell_ids = only_cells

    input_list = neuroml.InputList(id=id,
                                   component=input_comp_id,
                                   populations=population.id)
    count = 0
    for cell_id in cell_ids:
        for i in range(number_per_cell):
            
            segment_id = -1
            if len(segment_ids)==1:
                segment_id = segment_ids[0]
            else:
                segment_id = segment_ids[i]
            
            fraction_along = -1
            if len(fraction_alongs)==1:
                fraction_along = fraction_alongs[0]
            else:
                fraction_along = fraction_alongs[i]
                
            if fraction_along<0 or fraction_along>1:
                error = "Error! Attribute fraction_along should be >=0 and <=1" 
                opencortex.print_comment_v(error)
                raise Exception(error)
            
                
            input = neuroml.Input(id=count, 
                                  target="../%s/%i/%s" % (population.id, cell_id, population.component),
                                  segment_id=segment_id,
                                  fraction_along=fraction_along,
                                  destination="synapses")  
            input_list.input.append(input)
            count += 1

    if count > 0:                 
        net.input_lists.append(input_list)

    return input_list




##############################################################################################

def add_targeted_inputs_to_population(net, id, population, input_comp_id, segment_group, number_per_cell=1, all_cells=False, only_cells=None):
    
    """
    Add current input to the specified population. Attributes:
    
    `net`
        reference to the network object previously created

    `id` 
        id of the <inputList> to be created
        
    `population` 
        the <population> to be targeted
        
    `input_comp_id` 
        id of the component to be used for the input (e.g. added with add_pulse_generator())
        
    `segment_group`
        which segment group on the target cells to limit input locations to 

        
    `number_per_cell`
        How many inputs to apply to each cell of the population. Default 1
        
    `all_cells`
        whether to target all cells. Default False
        
    `only_cells`
        which specific cells to target. List of ids. Default None
        
        
    """

    if all_cells and only_cells is not None:
        error = "Error! Method opencortex.build.%s() called with both arguments all_cells and only_cells set!" % sys._getframe().f_code.co_name
        opencortex.print_comment_v(error)
        raise Exception(error)

    cell_ids = []
    
    target_cell = oc_build.cell_ids_vs_nml_docs[population.component].get_by_id(population.component)

    target_segs = oc_build.extract_seg_ids(target_cell,
                                           [segment_group],
                                           "segGroups")

    seg_target_dict = oc_build.make_target_dict(target_cell, target_segs)
    
    subset_dict = {segment_group: number_per_cell}
    
    if all_cells:
        cell_ids = range(population.size)
    if only_cells is not None:
        if only_cells == []:
            return
        cell_ids = only_cells

    input_list = neuroml.InputList(id=id,
                                   component=input_comp_id,
                                   populations=population.id)
    count = 0
    for cell_id in cell_ids:
        
        target_seg_array, target_fractions = oc_build.get_target_segments(seg_target_dict, subset_dict)
    
        for i in range(number_per_cell):
            input = neuroml.Input(id=count, 
                                  target="../%s/%i/%s" % (population.id, cell_id, population.component),
                                  segment_id=target_seg_array[i],
                                  fraction_along=target_fractions[i],
                                  destination="synapses")  
            input_list.input.append(input)
            count += 1

    if count > 0:                 
        net.input_lists.append(input_list)

    return input_list



##############################################################################################

def generate_network(reference, network_seed=1234, temperature='32degC'):
    
    """
    Generate a network which will contain populations, projections, etc. Arguments:
    
    `reference`
        the reference to use as the id for the network
        
    `network_seed`
        optional, will be used for random elements of the network, e.g. placement of cells in 3D
        
    `temperature`
        optional, will be specified in network and used in temperature dependent elements, e.g. ion channels with Q10. Default: 32degC
        
    """

    del oc_build.all_included_files[:]
    oc_build.all_cells.clear()

    nml_doc = neuroml.NeuroMLDocument(id='%s' % reference)

    random.seed(network_seed)

    nml_doc.properties.append(neuroml.Property("Network seed", network_seed))

    # Create network
    network = neuroml.Network(id='%s' % reference, type='networkWithTemperature', temperature=temperature)
    nml_doc.networks.append(network)

    opencortex.print_comment_v("Created NeuroMLDocument containing a network with id: %s" % reference)

    return nml_doc, network


##############################################################################################

def save_network(nml_doc, nml_file_name, validate=True, comment=True, format='xml', max_memory=None):
    
    """
    Save the contents of the built NeuroML document, including the network to the file specified by `nml_file_name`
    """

    info = "\n\nThis NeuroML 2 file was generated by OpenCortex v%s using: \n" % (opencortex.__version__)
    info += "    libNeuroML v%s\n" % (neuroml.__version__)
    info += "    pyNeuroML v%s\n\n    " % (pyneuroml.__version__)

    if nml_doc.notes:
        nml_doc.notes += info
    else:
        nml_doc.notes = info

    if format == 'xml':
        writers.NeuroMLWriter.write(nml_doc, nml_file_name)
    elif format == 'xml_hdf5':
        writers.NeuroMLHdf5Writer.write_xml_and_hdf5(nml_doc, nml_file_name, '%s.h5' % nml_file_name)
    elif format == 'hdf5':
        writers.NeuroMLHdf5Writer.write(nml_doc, nml_file_name)

    opencortex.print_comment_v("Saved NeuroML with id: %s to %s" % (nml_doc.id, nml_file_name))

    if validate:

        from pyneuroml.pynml import validate_neuroml2

        passed = validate_neuroml2(nml_file_name, max_memory=max_memory)

        if passed:
            opencortex.print_comment_v("Generated NeuroML file is valid")
        else:
            opencortex.print_comment_v("Generated NeuroML file is NOT valid!")


##############################################################################################

def generate_lems_simulation(nml_doc, 
                             network, 
                             nml_file_name, 
                             duration, 
                             dt, 
                             target_dir='.',
                             include_extra_lems_files=[],
                             gen_plots_for_all_v=True,
                             plot_all_segments=False,
                             gen_plots_for_quantities={}, #  Dict with displays vs lists of quantity paths
                             gen_plots_for_only_populations=[], #  List of populations, all pops if = []
                             gen_saves_for_all_v=True,
                             save_all_segments=False,
                             gen_saves_for_only_populations=[], #  List of populations, all pops if = []
                             gen_saves_for_quantities={}, #  Dict with file names vs lists of quantity paths
                             gen_spike_saves_for_all_somas=False,
                             spike_time_format='ID_TIME',
                             lems_file_name=None,
                             simulation_seed=12345):
                                 
    """
    Generate a LEMS simulation file with which to run simulations of the network. Generated LEMS files can be run with
    jNeuroML (or converted to simulator specific formats, e.g. NEURON, and run)
    """

    if not lems_file_name:
        lems_file_name = "LEMS_%s.xml" % network.id

    include_extra_lems_files.extend(oc_build.all_included_files)

    pyneuroml.lems.generate_lems_file_for_neuroml("Sim_%s" % network.id, 
                                                  nml_file_name, 
                                                  network.id, 
                                                  duration, 
                                                  dt, 
                                                  lems_file_name,
                                                  target_dir,
                                                  include_extra_files=include_extra_lems_files,
                                                  gen_plots_for_all_v=gen_plots_for_all_v,
                                                  plot_all_segments=plot_all_segments,
                                                  gen_plots_for_quantities=gen_plots_for_quantities, 
                                                  gen_plots_for_only_populations=gen_plots_for_only_populations, 
                                   gen_saves_for_all_v=gen_saves_for_all_v,
                                   save_all_segments=save_all_segments,
                                   gen_saves_for_only_populations=gen_saves_for_only_populations,
                                   gen_saves_for_quantities=gen_saves_for_quantities,
                                   gen_spike_saves_for_all_somas=gen_spike_saves_for_all_somas,
                                   spike_time_format=spike_time_format,
                                   seed=simulation_seed)

    del include_extra_lems_files[:]

    return lems_file_name


##############################################################################################

def simulate_network(lems_file_name,
                     simulator,
                     max_memory='400M',
                     nogui=True,
                     load_saved_data=False,
                     reload_events=False,
                     plot=False,
                     verbose=True):

    """
    Run a simulation of the LEMS file `lems_file_name` using target platform `simulator`
    """

    if simulator == "jNeuroML":
       results = pynml.run_lems_with_jneuroml(lems_file_name, max_memory=max_memory, nogui=nogui, load_saved_data=load_saved_data, reload_events=reload_events, plot=plot, verbose=verbose)
    elif simulator == "jNeuroML_NEURON":
       results = pynml.run_lems_with_jneuroml_neuron(lems_file_name, max_memory=max_memory, nogui=nogui, load_saved_data=load_saved_data, reload_events=reload_events, plot=plot, verbose=verbose)
    elif simulator == "jNeuroML_NetPyNE":
       results = pynml.run_lems_with_jneuroml_netpyne(lems_file_name, max_memory=max_memory, nogui=nogui, load_saved_data=load_saved_data, reload_events=reload_events, plot=plot, verbose=verbose)
    else:
        raise Exception("Simulator %s not yet supported"%simulator)

    if load_saved_data:
        return results
    


