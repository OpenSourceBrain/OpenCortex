
import opencortex.build as oc

population_size = 3

nml_doc, network = oc.generate_network("SimpleNet")

oc.add_cell_to_network(nml_doc,'../NeuroML2/prototypes/izhikevich/RS.cell.nml')

pop = oc.add_population_in_rectangular_region(network,
                                              'RS_pop',
                                              'RS',
                                              population_size,
                                              0,0,0,
                                              100,100,100)
                                              
syn = oc.add_exp_two_syn(nml_doc, id="syn0", gbase="2nS",
                             erev="0mV",
                             tau_rise="0.5ms",
                             tau_decay="10ms")


oc.save_network(nml_doc, '%s.net.nml'%network.id)
                                              
