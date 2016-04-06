
import opencortex.build as oc


nml_doc, network = oc.generate_network("IClamps")

oc.add_cell_prototype(nml_doc, '../NeuroML2/prototypes/izhikevich/RS.cell.nml')
oc.add_cell_prototype(nml_doc, '../NeuroML2/prototypes/acnet2/pyr_4_sym_soma.cell.nml')

popIzh = oc.add_single_cell_population(network,
                                     'popIzh',
                                     'RS')

popHH = oc.add_single_cell_population(network,
                                     'popHH',
                                     'pyr_4_sym_soma',
                                     z=100)
                                     
pgIzh = oc.add_pulse_generator(nml_doc,
                       id="pgIzh",
                       delay="100ms",
                       duration="300ms",
                       amplitude="0.7nA")
                                     
pgHH = oc.add_pulse_generator(nml_doc,
                       id="pgHH",
                       delay="100ms",
                       duration="300ms",
                       amplitude="0.7nA")
                                              
                                   
oc.add_inputs_to_population(network,
                            "Stim0",
                            popIzh,
                            pgIzh.id,
                            all_cells=True)
                        
oc.add_inputs_to_population(network,
                            "Stim1",
                            popHH,
                            pgHH.id,
                            all_cells=True)

nml_file_name = '%s.net.nml'%network.id
oc.save_network(nml_doc, nml_file_name, validate=True)

oc.generate_lems_simulation(nml_doc, 
                            network, 
                            nml_file_name, 
                            duration =      500, 
                            dt =            0.025)
                                              
