'''
Generates a complex NeuroML 2 file with many cells, populations and inputs
for testing purposes
'''

import opencortex.build as oc


nml_doc, network = oc.generate_network("Complex")

scale = 0.1
min_pop_size = 3

def scale_pop_size(baseline):
    return max(min_pop_size, int(baseline*scale))

#####   Cells

oc.add_cell_prototype(nml_doc, '../NeuroML2/prototypes/izhikevich/RS.cell.nml')
oc.add_cell_prototype(nml_doc, '../NeuroML2/prototypes/iaf/iaf.cell.nml')
oc.add_cell_prototype(nml_doc, '../NeuroML2/prototypes/iaf/iafRef.cell.nml')
oc.add_cell_prototype(nml_doc, '../NeuroML2/prototypes/acnet2/pyr_4_sym_soma.cell.nml')

xDim = 500
yDim = 100
zDim = 500
offset = 0


#####   Synapses

synAmpa1 = oc.add_exp_two_syn(nml_doc, id="synAmpa1", gbase="1nS",
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


          
#####   Inputs

oc.add_inputs_to_population(network, "Stim0",
                            popIzh, pfs100.id,
                            all_cells=True)

oc.add_inputs_to_population(network, "Stim1",
                            popIaf, pfs200.id,
                            all_cells=True)

oc.add_inputs_to_population(network, "Stim3",
                            popIafRef, pfs200.id,
                            all_cells=True)


nml_file_name = '%s.net.nml'%network.id
oc.save_network(nml_doc, nml_file_name, validate=True)

oc.generate_lems_simulation(nml_doc, 
                            network, 
                            nml_file_name, 
                            duration =      500, 
                            dt =            0.025)
                                              
