'''
Generates a NeuroML 2 file with many types of cells, populations and inputs
for testing purposes
'''

import opencortex.build as oc
import sys


min_pop_size = 3

def scale_pop_size(baseline, scale):
    return max(min_pop_size, int(baseline*scale))



def generate(reference = "Balanced",
             num_bbp =1,
             scalePops = 1,
             scalex=1,
             scaley=1,
             scalez=1,
             connections=True,
             global_delay = 0,
             format='xml'):

    num_exc = scale_pop_size(80,scalePops)
    num_inh = scale_pop_size(40,scalePops)
    
    nml_doc, network = oc.generate_network(reference)

    oc.add_cell_and_channels(nml_doc, 'AllenInstituteCellTypesDB_HH/HH_464198958.cell.nml','HH_464198958')
    oc.add_cell_and_channels(nml_doc, 'AllenInstituteCellTypesDB_HH/HH_471141261.cell.nml','HH_471141261')
    
    if num_bbp>0:
        oc.add_cell_and_channels(nml_doc, 'BlueBrainProject_NMC/cADpyr229_L23_PC_5ecbf9b163_0_0.cell.nml', 'cADpyr229_L23_PC_5ecbf9b163_0_0')

    xDim = 400*scalex
    yDim = 500*scaley
    zDim = 300*scalez

    xs = -200
    ys = -150
    zs = 100

    #####   Synapses

    synAmpa1 = oc.add_exp_two_syn(nml_doc, id="synAmpa1", gbase="1nS",
                             erev="0mV", tau_rise="0.5ms", tau_decay="5ms")

    synGaba1 = oc.add_exp_two_syn(nml_doc, id="synGaba1", gbase="2nS",
                             erev="-80mV", tau_rise="1ms", tau_decay="20ms")

    #####   Input types


    pfs1 = oc.add_poisson_firing_synapse(nml_doc,
                                       id="psf1",
                                       average_rate="150 Hz",
                                       synapse_id=synAmpa1.id)


    #####   Populations

    popExc = oc.add_population_in_rectangular_region(network,
                                                  'popExc',
                                                  'HH_464198958',
                                                  num_exc,
                                                  xs,ys,zs,
                                                  xDim,yDim,zDim)

    popInh = oc.add_population_in_rectangular_region(network,
                                                  'popInh',
                                                  'HH_471141261',
                                                  num_inh,
                                                  xs,ys,zs,
                                                  xDim,yDim,zDim)
    if num_bbp == 1:
        popBBP = oc.add_single_cell_population(network,
                                             'popBBP',
                                             'cADpyr229_L23_PC_5ecbf9b163_0_0',
                                             z=200)
    elif num_bbp > 1:

        popBBP = oc.add_population_in_rectangular_region(network,
                                                      'popBBP',
                                                      'cADpyr229_L23_PC_5ecbf9b163_0_0',
                                                      num_bbp,
                                                      xs,ys,zs,
                                                      xDim,yDim,zDim)


    #####   Projections

    total_conns = 0
    if connections:
        proj = oc.add_probabilistic_projection(network, "proj0",
                                        popExc, popExc,
                                        synAmpa1.id, 0.3, delay = global_delay)
        total_conns += len(proj.connection_wds)

        proj = oc.add_probabilistic_projection(network, "proj1",
                                        popExc, popInh,
                                        synAmpa1.id, 0.5, delay = global_delay)
        total_conns += len(proj.connection_wds)

        proj = oc.add_probabilistic_projection(network, "proj3",
                                        popInh, popExc,
                                        synGaba1.id, 0.7, delay = global_delay)
        total_conns += len(proj.connection_wds)

        proj = oc.add_probabilistic_projection(network, "proj4",
                                        popInh, popInh,
                                        synGaba1.id, 0.5, delay = global_delay)
        total_conns += len(proj.connection_wds)



        if num_bbp>0:
            proj = oc.add_probabilistic_projection(network, "proj5",
                                            popExc, popBBP,
                                            synAmpa1.id, 0.5, delay = global_delay)
                                        
        total_conns += len(proj.connection_wds)

    #####   Inputs

    oc.add_inputs_to_population(network, "Stim0",
                                popExc, pfs1.id,
                                all_cells=True)



    #####   Save NeuroML and LEMS Simulation files      
    
    if num_bbp != 1:
        new_reference = 'Balanced_%scells_%sconns'%(num_bbp+num_exc+num_inh,total_conns)
        network.id = new_reference
        nml_doc.id = new_reference

    nml_file_name = '%s.net.%s'%(network.id,'nml.h5' if format == 'hdf5' else 'nml')
    oc.save_network(nml_doc, 
                    nml_file_name, 
                    validate=(format=='xml'),
                    format = format)

    if format=='xml':
        lems_file_name = oc.generate_lems_simulation(nml_doc, network, 
                                nml_file_name, 
                                duration =      1000, 
                                dt =            0.025)
    else:
        lems_file_name = None
                                
    return nml_doc, nml_file_name, lems_file_name
                               

if __name__ == '__main__':
    
    if '-all' in sys.argv:
        generate()
        generate(num_bbp =10,
             scalePops = 5,
             scalex=2,
             scalez=2,
             connections=False)
        
        generate(num_bbp =10,
             scalePops = 5,
             scalex=2,
             scalez=2)
             
    if '-test' in sys.argv:
        
        generate(num_bbp =0,
             scalePops = 1,
             scalex=2,
             scalez=2,
             global_delay = 2)
    else:
        generate()