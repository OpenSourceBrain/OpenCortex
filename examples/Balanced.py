'''
Generates a NeuroML 2 file with many types of cells, populations and inputs
for testing purposes
'''

import opencortex.build as oc


nml_doc, network = oc.generate_network("Balanced")

scale = .4
min_pop_size = 3

def scale_pop_size(baseline):
    return max(min_pop_size, int(baseline*scale))


oc.add_cell_and_channels(nml_doc, '../NeuroML2/prototypes/AllenInstituteCellTypesDB_HH/HH_464198958.cell.nml','HH_464198958')
oc.add_cell_and_channels(nml_doc, '../NeuroML2/prototypes/AllenInstituteCellTypesDB_HH/HH_471141261.cell.nml','HH_471141261')
oc.add_cell_and_channels(nml_doc, '../NeuroML2/prototypes/BlueBrainProject_NMC/cADpyr229_L23_PC_5ecbf9b163_0_0.cell.nml', 'cADpyr229_L23_PC_5ecbf9b163_0_0')


xDim = 400
yDim = 500
zDim = 300

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
                                              scale_pop_size(80),
                                              xs,ys,zs,
                                              xDim,yDim,zDim)

popInh = oc.add_population_in_rectangular_region(network,
                                              'popInh',
                                              'HH_471141261',
                                              scale_pop_size(40),
                                              xs,ys,zs,
                                              xDim,yDim,zDim)

popBBP = oc.add_single_cell_population(network,
                                     'popBBP',
                                     'cADpyr229_L23_PC_5ecbf9b163_0_0',
                                     z=200)


#####   Projections


oc.add_probabilistic_projection(network, "proj0",
                                popExc, popExc,
                                synAmpa1.id, 0.3)

oc.add_probabilistic_projection(network, "proj1",
                                popExc, popInh,
                                synAmpa1.id, 0.5)

oc.add_probabilistic_projection(network, "proj3",
                                popInh, popExc,
                                synGaba1.id, 0.7)

oc.add_probabilistic_projection(network, "proj4",
                                popInh, popInh,
                                synGaba1.id, 0.5)



oc.add_probabilistic_projection(network, "proj5",
                                popExc, popBBP,
                                synAmpa1.id, 1)
          
#####   Inputs

oc.add_inputs_to_population(network, "Stim0",
                            popExc, pfs1.id,
                            all_cells=True)
                            
            

#####   Save NeuroML and LEMS Simulation files               

nml_file_name = '%s.net.nml'%network.id
oc.save_network(nml_doc, nml_file_name, validate=True)

oc.generate_lems_simulation(nml_doc, network, 
                            nml_file_name, 
                            duration =      1000, 
                            dt =            0.025)
