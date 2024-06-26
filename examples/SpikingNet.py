
import opencortex.core as oc

population_size0 = 10
population_size1 = 10

nml_doc, network = oc.generate_network("SpikingNet")

oc.include_opencortex_cell(nml_doc, 'izhikevich/RS.cell.nml')

pop_pre = oc.add_population_in_rectangular_region(network,
                                              'pop_pre',
                                              'RS',
                                              population_size0,
                                              0,0,0,
                                              100,100,100,
                                              color='.8 0 0')
import neuroml
pop_pre.properties.append(neuroml.Property('radius',10))

pop_post = oc.add_population_in_rectangular_region(network,
                                              'pop_post',
                                              'RS',
                                              population_size1,
                                              0,100,0,
                                              100,200,100,
                                              color='0 0 .8')
                                              
pop_post.properties.append(neuroml.Property('radius',10))
                                              
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

duration =      1000
dt =            0.01
nml_file_name = '%s.net.nml'%network.id

oc.save_network(nml_doc, nml_file_name, validate=True)

oc.generate_lems_simulation(nml_doc, 
                            network, 
                            nml_file_name, 
                            duration = duration, 
                            dt =       dt,
                            report_file_name='report.spiking.txt')
                                              
nml_file_name = '%s.net.nml.h5'%network.id
target_dir='HDF5/'
oc.save_network(nml_doc, nml_file_name, validate=False, target_dir=target_dir, format='hdf5')

oc.generate_lems_simulation(nml_doc, 
                            network, 
                            target_dir+nml_file_name, 
                            duration, 
                            dt,
                            target_dir=target_dir,
                            report_file_name='report.spiking.h5.txt')
