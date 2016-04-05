
import opencortex.build as oc

population_size0 = 10
population_size1 = 10

nml_doc, network = oc.generate_network("SpikingNet")

oc.add_cell_prototype(nml_doc, '../NeuroML2/prototypes/izhikevich/RS.cell.nml')

pop0 = oc.add_population_in_rectangular_region(network,
                                              'pop0',
                                              'RS',
                                              population_size0,
                                              0,0,0,
                                              100,100,100)

pop1 = oc.add_population_in_rectangular_region(network,
                                              'pop1',
                                              'RS',
                                              population_size1,
                                              0,100,0,
                                              100,200,100)
                                              
syn = oc.add_exp_two_syn(nml_doc, 
                         id="syn0", 
                         gbase="1nS",
                         erev="0mV",
                         tau_rise="0.5ms",
                         tau_decay="10ms")
                             
pfs = oc.add_poisson_firing_synapse(nml_doc,
                                   id="poissonFiringSyn",
                                   average_rate="150 Hz",
                                   synapse_id=syn.id)
                                   
oc.add_inputs_to_population(network,
                            "Stim0",
                            pop0,
                            pfs.id,
                            all_cells=True)
                            
oc.add_probabilistic_projection(network,
                                "proj0",
                                pop0,
                                pop1,
                                syn.id,
                                0.5)

nml_file_name = '%s.net.nml'%network.id
oc.save_network(nml_doc, nml_file_name, validate=True)

oc.generate_lems_simulation(nml_doc, 
                            network, 
                            nml_file_name, 
                            duration =      1000, 
                            dt =            0.05)
                                              
