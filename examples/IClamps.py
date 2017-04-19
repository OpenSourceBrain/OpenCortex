
import opencortex.core as oc

nml_doc, network = oc.generate_network("IClamps")

oc.include_opencortex_cell(nml_doc, 'izhikevich/RS.cell.nml')
oc.include_opencortex_cell(nml_doc, 'acnet2/pyr_4_sym_soma.cell.nml')
#oc.include_opencortex_cell(nml_doc, '../NeuroML2/prototypes/BlueBrainProject_NMC/cADpyr229_L23_PC_5ecbf9b163_0_0.cell.nml')

popIzh = oc.add_single_cell_population(network,
                                     'popIzh',
                                     'RS')

popHH = oc.add_single_cell_population(network,
                                     'popHH',
                                     'pyr_4_sym_soma',
                                     z=100)
'''
popBBP = oc.add_single_cell_population(network,
                                     'popBBP',
                                     'cADpyr229_L23_PC_5ecbf9b163_0_0',
                                     z=200)'''
                                     
pgIzh = oc.add_pulse_generator(nml_doc,
                       id="pgIzh",
                       delay="100ms",
                       duration="300ms",
                       amplitude="0.5nA")
                                     
pgHH = oc.add_pulse_generator(nml_doc,
                       id="pgHH",
                       delay="100ms",
                       duration="300ms",
                       amplitude="0.7nA")
'''                                     
pgBBP = oc.add_pulse_generator(nml_doc,
                       id="pgBBP",
                       delay="100ms",
                       duration="300ms",
                       amplitude="0.7nA")'''
                                              
                                   
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
'''                        
oc.add_inputs_to_population(network,
                            "Stim2",
                            popBBP,
                            pgBBP.id,
                            all_cells=True)'''

nml_file_name = '%s.net.nml'%network.id
oc.save_network(nml_doc, nml_file_name, validate=True)

oc.generate_lems_simulation(nml_doc, 
                            network, 
                            nml_file_name, 
                            duration =      500, 
                            dt =            0.005)
                                              
