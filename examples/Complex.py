'''
Generates a complex NeuroML 2 file with many types of cells, populations and inputs
for testing purposes
'''

import opencortex.build as oc


nml_doc, network = oc.generate_network("Complex")

scale = 0.01
min_pop_size = 3

def scale_pop_size(baseline):
    return max(min_pop_size, int(baseline*scale))

#####   Cells

#oc.add_cell_prototype(nml_doc, 'izhikevich/Izh_471141261.cell.nml')
oc.add_cell_and_channels(nml_doc, 'izhikevich/RS.cell.nml','RS')
oc.add_cell_and_channels(nml_doc, 'iaf/iaf.cell.nml','iaf')
oc.add_cell_and_channels(nml_doc, 'iaf/iafRef.cell.nml','iafRef')
oc.add_cell_and_channels(nml_doc, 'acnet2/pyr_4_sym_soma.cell.nml','pyr_4_sym_soma')
oc.add_cell_and_channels(nml_doc, 'acnet2/pyr_4_sym.cell.nml','pyr_4_sym')

xDim = 500
yDim = 100
zDim = 500
offset = 0


#####   Synapses

synAmpa1 = oc.add_exp_two_syn(nml_doc, id="synAmpa1", gbase="1nS",
                         erev="0mV", tau_rise="0.5ms", tau_decay="10ms")
                         
synAmpa2 = oc.add_exp_two_syn(nml_doc, id="synAmpa2", gbase="2nS",
                         erev="0mV", tau_rise="0.5ms", tau_decay="10ms")

synGaba1 = oc.add_exp_two_syn(nml_doc, id="synGaba1", gbase="1nS",
                         erev="-70mV", tau_rise="2ms", tau_decay="20ms")

#####   Input types

                             
pfs100 = oc.add_poisson_firing_synapse(nml_doc,
                                   id="poissonFiringSyn100",
                                   average_rate="100 Hz",
                                   synapse_id=synAmpa1.id)
                             
pfs200 = oc.add_poisson_firing_synapse(nml_doc,
                                   id="poissonFiringSyn200",
                                   average_rate="200 Hz",
                                   synapse_id=synAmpa1.id)
                             
pfsStrong = oc.add_poisson_firing_synapse(nml_doc,
                                   id="poissonFiringSynStrong",
                                   average_rate="200 Hz",
                                   synapse_id=synAmpa2.id)
                                   
#####   Populations

popIaf = oc.add_population_in_rectangular_region(network,
                                              'popIaf',
                                              'iaf',
                                              scale_pop_size(20),
                                              0,offset,0,
                                              xDim,yDim,zDim)
offset+=yDim

popIafRef = oc.add_population_in_rectangular_region(network,
                                              'popIafRef',
                                              'iafRef',
                                              scale_pop_size(20),
                                              0,offset,0,
                                              xDim,yDim,zDim)
offset+=yDim

popIzh = oc.add_population_in_rectangular_region(network,
                                              'popIzh',
                                              'RS',
                                              scale_pop_size(20),
                                              0,offset,0,
                                              xDim,yDim,zDim)
offset+=yDim

popPyrS = oc.add_population_in_rectangular_region(network,
                                              'popPyrS',
                                              'pyr_4_sym_soma',
                                              scale_pop_size(20),
                                              0,offset,0,
                                              xDim,yDim,zDim)
offset+=yDim

popPyr = oc.add_population_in_rectangular_region(network,
                                              'popPyr',
                                              'pyr_4_sym',
                                              scale_pop_size(20),
                                              0,offset,0,
                                              xDim,yDim,zDim)
offset+=yDim

#####   Projections


oc.add_probabilistic_projection(network,
                                "proj0",
                                popIaf,
                                popIzh,
                                synAmpa1.id,
                                0.5)
          
#####   Inputs

oc.add_inputs_to_population(network, "Stim0",
                            popIzh, pfsStrong.id,
                            all_cells=True)

oc.add_inputs_to_population(network, "Stim1",
                            popIaf, pfs200.id,
                            all_cells=True)

oc.add_inputs_to_population(network, "Stim2",
                            popIafRef, pfs200.id,
                            all_cells=True)

oc.add_inputs_to_population(network, "Stim3",
                            popPyrS, pfs100.id,
                            all_cells=True)

oc.add_inputs_to_population(network, "Stim4",
                            popPyr, pfs200.id,
                            all_cells=True)


#####   Save NeuroML and LEMS Simulation files

nml_file_name = '%s.net.nml'%network.id
oc.save_network(nml_doc, nml_file_name, validate=True)

oc.generate_lems_simulation(nml_doc, 
                            network, 
                            nml_file_name, 
                            duration =      500, 
                            dt =            0.025)
                                              
