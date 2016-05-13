#####################
### Subject to change without notice!!
#####################

import opencortex
import neuroml
import pyneuroml
import pyneuroml.lems

import neuroml.writers as writers
import neuroml.loaders as loaders

from pyneuroml import pynml
from pyneuroml.lems.LEMSSimulation import LEMSSimulation

import random
import sys
import os
import shutil

all_cells = {}
#all_included_on_cells = {}

all_included_files = []

def add_connection(projection, 
                   id, 
                   presynaptic_population, 
                   pre_cell_id, 
                   pre_seg_id, 
                   postsynaptic_population, 
                   post_cell_id, 
                   post_seg_id,
                   delay,
                   weight):

    connection = neuroml.ConnectionWD(id=id, \
                            pre_cell_id="../%s/%i/%s"%(presynaptic_population.id, pre_cell_id, presynaptic_population.component), \
                            pre_segment_id=pre_seg_id, \
                            pre_fraction_along=0.5,
                            post_cell_id="../%s/%i/%s"%(postsynaptic_population.id, post_cell_id, postsynaptic_population.component), \
                            post_segment_id=post_seg_id,
                            post_fraction_along=0.5,
                            delay = '%s ms'%delay,
                            weight = weight)

    projection.connection_wds.append(connection)
    

def add_probabilistic_projection(net, 
                                 prefix, 
                                 presynaptic_population, 
                                 postsynaptic_population, 
                                 synapse_id,  
                                 connection_probability,
                                 delay = 0,
                                 weight = 1):
    
    if presynaptic_population.size==0 or postsynaptic_population.size==0:
        return None

    proj = neuroml.Projection(id="%s_%s_%s"%(prefix,presynaptic_population.id, postsynaptic_population.id), 
                      presynaptic_population=presynaptic_population.id, 
                      postsynaptic_population=postsynaptic_population.id, 
                      synapse=synapse_id)


    count = 0

    for i in range(0, presynaptic_population.size):
        for j in range(0, postsynaptic_population.size):
            if i != j or presynaptic_population.id != postsynaptic_population.id:
                if connection_probability>= 1 or random.random() < connection_probability:
                    add_connection(proj, 
                                   count, 
                                   presynaptic_population, 
                                   i, 
                                   0, 
                                   postsynaptic_population, 
                                   j, 
                                   0,
                                   delay = delay,
                                   weight = weight)
                    count+=1

    net.projections.append(proj)

    return proj
    
    
def include_cell_prototype(nml_doc,cell_nml2_path):
    
    nml_doc.includes.append(neuroml.IncludeType(cell_nml2_path)) 
    
# Helper method which will be made redundant with a better generated Python API...
def _get_cells_of_all_known_types(nml_doc):
    
    all_cells = []
    all_cells.extend(nml_doc.cells)
    all_cells.extend(nml_doc.izhikevich_cells)
    all_cells.extend(nml_doc.izhikevich2007_cells)
    all_cells.extend(nml_doc.iaf_cells)
    all_cells.extend(nml_doc.iaf_ref_cells)
    
    return all_cells

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
        
    
def _copy_to_dir_for_model(nml_doc,file_name):
    
    dir_for_model = nml_doc.id
    if not os.path.isdir(dir_for_model):
        os.mkdir(dir_for_model)
    
    shutil.copy(file_name, dir_for_model)
    
    
def add_cell_and_channels(nml_doc,cell_nml2_path, cell_id):
    
    nml2_doc_cell = pynml.read_neuroml2_file(cell_nml2_path, include_includes=False)
    
    for cell in _get_cells_of_all_known_types(nml2_doc_cell):
        if cell.id == cell_id:
            all_cells[cell_id] = cell
            
            _copy_to_dir_for_model(nml_doc,cell_nml2_path)
            new_file = '%s/%s.cell.nml'%(nml_doc.id,cell_id)
            nml_doc.includes.append(neuroml.IncludeType(new_file)) 
            all_included_files.append(new_file)
            
            for included in nml2_doc_cell.includes:
                #Todo replace... quick & dirty...
                old_loc = '%s/%s'%(os.path.dirname(os.path.abspath(cell_nml2_path)), included.href)
                print old_loc
                _copy_to_dir_for_model(nml_doc,old_loc)
                new_loc = '%s/%s'%(nml_doc.id,included.href)
                nml_doc.includes.append(neuroml.IncludeType(new_loc))
                all_included_files.append(new_loc)

    
    
def add_exp_two_syn(nml_doc, id, gbase, erev, tau_rise, tau_decay):
    # Define synapse
    syn0 = neuroml.ExpTwoSynapse(id=id, gbase=gbase,
                                 erev=erev,
                                 tau_rise=tau_rise,
                                 tau_decay=tau_decay)
                                 
    nml_doc.exp_two_synapses.append(syn0)
    
    return syn0

def add_poisson_firing_synapse(nml_doc, id, average_rate, synapse_id):

    pfs = neuroml.PoissonFiringSynapse(id=id,
                                       average_rate=average_rate,
                                       synapse=synapse_id, 
                                       spike_target="./%s"%synapse_id)
                                       
    nml_doc.poisson_firing_synapses.append(pfs)

    return pfs

def add_pulse_generator(nml_doc, id, delay, duration, amplitude):

    pg = neuroml.PulseGenerator(id=id,
                                delay=delay,
                                duration=duration,
                                amplitude=amplitude)
                                       
    nml_doc.pulse_generators.append(pg)

    return pg
    
    
def add_single_cell_population(net, pop_id, cell_id, x=0, y=0, z=0, color=None):
    
    pop = neuroml.Population(id=pop_id, component=cell_id, type="populationList", size=1)
    if color is not None:
        pop.properties.append(Property("color",color))
    net.populations.append(pop)

    inst = neuroml.Instance(id=0)
    pop.instances.append(inst)
    inst.location = neuroml.Location(x=x, y=y, z=z)

    return pop
    
    
def add_population_in_rectangular_region(net, pop_id, cell_id, size, x_min, y_min, z_min, x_size, y_size, z_size, color=None):
    
    pop = neuroml.Population(id=pop_id, component=cell_id, type="populationList", size=size)
    if color is not None:
        pop.properties.append(Property("color",color))
    net.populations.append(pop)

    for i in range(0, size) :
            index = i
            inst = neuroml.Instance(id=index)
            pop.instances.append(inst)
            inst.location = neuroml.Location(x=str(x_min +(x_size)*random.random()), y=str(y_min +(y_size)*random.random()), z=str(z_min+(z_size)*random.random()))
    
    return pop

def add_inputs_to_population(net, id, population, input_comp_id, all_cells=False, only_cells=None):
    
    if all_cells and only_cells is not None:
        opencortex.print_comment_v("Error! Method opencortex.build.%s() called with both arguments all_cells and only_cells set!"%sys._getframe().f_code.co_name)
        exit(-1)
        
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
        input = neuroml.Input(id=count, 
                      target="../%s/%i/%s"%(population.id, cell_id, population.component), 
                      destination="synapses")  
        input_list.input.append(input)
        count+=1
        
                         
    net.input_lists.append(input_list)
    
    return input_list
    

def generate_network(reference, seed=1234):
    
    nml_doc = neuroml.NeuroMLDocument(id='%s'%reference)
    
    random.seed(seed)
    
    nml_doc.properties.append(neuroml.Property("Python random seed",seed))
    
    # Create network
    network = neuroml.Network(id='%s'%reference)
    nml_doc.networks.append(network)

    opencortex.print_comment_v("Created NeuroMLDocument containing a network with id: %s"%reference)
    
    return nml_doc, network


def save_network(nml_doc, nml_file_name, validate=True, comment=True):

    info = "\n\nThis NeuroML 2 file was generated by OpenCortex v%s using: \n"%(opencortex.__version__)
    info += "    libNeuroML v%s\n"%(neuroml.__version__)
    info += "    pyNeuroML v%s\n\n    "%(pyneuroml.__version__)
    
    if nml_doc.notes:
        nml_doc.notes += info
    else:
        nml_doc.notes = info
    
    writers.NeuroMLWriter.write(nml_doc, nml_file_name)
    
    opencortex.print_comment_v("Saved NeuroML with id: %s to %s"%(nml_doc.id, nml_file_name))
    
    if validate:
        from pyneuroml.pynml import validate_neuroml2

        passed = validate_neuroml2(nml_file_name)
        
        if passed:
            opencortex.print_comment_v("Generated NeuroML file is valid")
        else:
            opencortex.print_comment_v("Generated NeuroML file is NOT valid!")
            
            
def generate_lems_simulation(nml_doc, 
                             network, 
                             nml_file_name, 
                             duration, 
                             dt, 
                             target_dir = '.',
                             include_extra_files = [],
                             gen_plots_for_all_v = True,
                             plot_all_segments = False,
                             gen_plots_for_quantities = {},   #  Dict with displays vs lists of quantity paths
                             gen_plots_for_only_populations = [],   #  List of populations, all pops if = []
                             gen_saves_for_all_v = True,
                             save_all_segments = False,
                             gen_saves_for_only_populations = [],  #  List of populations, all pops if = []
                             gen_saves_for_quantities = {},   #  Dict with file names vs lists of quantity paths
                             seed=12345):
                                 
    lems_file_name = "LEMS_%s.xml"%network.id
    
    include_extra_files.extend(all_included_files)
    
    pyneuroml.lems.generate_lems_file_for_neuroml("Sim_%s"%network.id, 
                                   nml_file_name, 
                                   network.id, 
                                   duration, 
                                   dt, 
                                   lems_file_name,
                                   target_dir,
                                   include_extra_files = include_extra_files,
                                   gen_plots_for_all_v = gen_plots_for_all_v,
                                   plot_all_segments = plot_all_segments,
                                   gen_plots_for_quantities = gen_plots_for_quantities, 
                                   gen_plots_for_only_populations = gen_plots_for_only_populations,  
                                   gen_saves_for_all_v = gen_saves_for_all_v,
                                   save_all_segments = save_all_segments,
                                   gen_saves_for_only_populations = gen_saves_for_only_populations,
                                   gen_saves_for_quantities = gen_saves_for_quantities,
                                   seed=seed)

    return lems_file_name