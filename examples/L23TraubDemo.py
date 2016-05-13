
import opencortex.build as oc

population_size0 = 2
population_size1 = 2

nml_doc, network = oc.generate_network("L23TraubDemo")

oc.add_cell_and_channels(nml_doc, '../NeuroML2/prototypes/Thalamocortical/L23PyrRS.cell.nml','L23PyrRS')
oc.add_cell_and_channels(nml_doc, '../NeuroML2/prototypes/Thalamocortical/SupBasket.cell.nml','SupBasket')

pop_pre = oc.add_population_in_rectangular_region(network,
                                              'pop_pre',
                                              'L23PyrRS',
                                              population_size0,
                                              0,0,0,
                                              100,100,100)

pop_post = oc.add_population_in_rectangular_region(network,
                                              'pop_post',
                                              'SupBasket',
                                              population_size1,
                                              0,100,0,
                                              100,200,100)
                                              
syn0 = oc.add_exp_two_syn(nml_doc, 
                         id="syn0", 
                         gbase="1nS",
                         erev="0mV",
                         tau_rise="0.5ms",
                         tau_decay="10ms")
             
syn1 = oc.add_exp_two_syn(nml_doc, 
                         id="syn1", 
                         gbase="2nS",
                         erev="0mV",
                         tau_rise="1ms",
                         tau_decay="15ms")
                             
pfs = oc.add_poisson_firing_synapse(nml_doc,
                                   id="poissonFiringSyn",
                                   average_rate="150 Hz",
                                   synapse_id=syn0.id)
                                   
oc.add_inputs_to_population(network,
                            "Stim0",
                            pop_pre,
                            pfs.id,
                            all_cells=True)
                            
oc.add_probabilistic_projection(network,
                                "proj0",
                                pop_pre,
                                pop_post,
                                syn1.id,
                                0.3,
                                weight=0.05,
                                delay=5)

nml_file_name = '%s.net.nml'%network.id
oc.save_network(nml_doc, nml_file_name, validate=True)

oc.generate_lems_simulation(nml_doc, 
                            network, 
                            nml_file_name, 
                            duration =      300, 
                            dt =            0.025)
                                              
