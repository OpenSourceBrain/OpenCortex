'''
Generates a deterministic NeuroML 2 file (not stochastic elements )with many 
types of cells, populations and inputs for testing exact spike times across 
simulations
'''

import opencortex.build as oc


nml_doc, network = oc.generate_network("Deterministic")


#####   Cells

oc.add_cell_and_channels(nml_doc, 'izhikevich/RS.cell.nml','RS')
oc.add_cell_and_channels(nml_doc, 'iaf/iaf.cell.nml','iaf')
oc.add_cell_and_channels(nml_doc, 'acnet2/pyr_4_sym_soma.cell.nml','pyr_4_sym_soma')

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

pg0 = oc.add_pulse_generator(nml_doc,
                       id="pg0",
                       delay="50ms",
                       duration="400ms",
                       amplitude="0.6nA")
                       
pg1 = oc.add_pulse_generator(nml_doc,
                       id="pg1",
                       delay="50ms",
                       duration="400ms",
                       amplitude="0.5nA")
                                   
#####   Populations

pop_iaf = oc.add_population_in_rectangular_region(network,
                                              'pop_iaf',
                                              'iaf',
                                              5,
                                              0,offset,0,
                                              xDim,yDim,zDim)
offset+=yDim

pop_rs = oc.add_population_in_rectangular_region(network,
                                              'pop_rs',
                                              'RS',
                                              5,
                                              0,offset,0,
                                              xDim,yDim,zDim)


#####   Projections


oc.add_probabilistic_projection(network,
                                "proj0",
                                pop_iaf,
                                pop_rs,
                                synAmpa1.id,
                                0.5)
          
#####   Inputs

oc.add_inputs_to_population(network, "Stim0",
                            pop_iaf, pg0.id,
                            only_cells=[i for i in [0,2] if i<pop_iaf.size])
                            
oc.add_inputs_to_population(network, "Stim1",
                            pop_iaf, pg1.id,
                            only_cells=[i for i in [3,4] if i<pop_iaf.size])



nml_file_name = '%s.net.nml'%network.id
oc.save_network(nml_doc, nml_file_name, validate=True)

oc.generate_lems_simulation(nml_doc, 
                            network, 
                            nml_file_name, 
                            duration =      500, 
                            dt =            0.005)
                                              
