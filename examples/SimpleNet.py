
import opencortex.build as oc

population_size = 3

nml_doc, network = oc.generate_network("SimpleNet")

oc.add_cell_and_channels(nml_doc, '../NeuroML2/prototypes/izhikevich/RS.cell.nml','RS')

pop = oc.add_population_in_rectangular_region(network,
                                              'RS_pop',
                                              'RS',
                                              population_size,
                                              0,0,0,
                                              100,100,100)
                                              
syn = oc.add_exp_two_syn(nml_doc, 
                         id="syn0", 
                         gbase="2nS",
                         erev="0mV",
                         tau_rise="0.5ms",
                         tau_decay="10ms")
                             
pfs = oc.add_poisson_firing_synapse(nml_doc,
                                   id="poissonFiringSyn",
                                   average_rate="50 Hz",
                                   synapse_id=syn.id)
                                   
oc.add_inputs_to_population(network,
                            "Stim0",
                            pop,
                            pfs.id,
                            all_cells=True)

nml_file_name = '%s.net.nml'%network.id
oc.save_network(nml_doc, nml_file_name, validate=True)

oc.generate_lems_simulation(nml_doc, 
                            network, 
                            nml_file_name, 
                            duration =      500, 
                            dt =            0.025)
                                              
