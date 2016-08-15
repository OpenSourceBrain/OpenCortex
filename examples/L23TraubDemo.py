
import opencortex.build as oc

import sys



def generate(reference = "L23TraubDemo",
             num_rs =2,
             num_bask =2,
             scalex=1,
             scaley=1,
             scalez=1,
             connections=True,
             global_delay = 0,
             duration = 300,
             format='xml'):


    nml_doc, network = oc.generate_network(reference)

    #oc.add_cell_and_channels(nml_doc, 'acnet2/pyr_4_sym.cell.nml','pyr_4_sym')
    oc.add_cell_and_channels(nml_doc, 'Thalamocortical/L23PyrRS.cell.nml','L23PyrRS')
    oc.add_cell_and_channels(nml_doc, 'Thalamocortical/SupBasket.cell.nml','SupBasket')
    
    xDim = 500*scalex
    yDim = 200*scaley
    zDim = 500*scalez

    pop_pre = oc.add_population_in_rectangular_region(network,
                                                  'pop_pre',
                                                  'L23PyrRS',
                                                  num_rs,
                                                  0,0,0,
                                                  xDim,yDim,zDim)

    pop_post = oc.add_population_in_rectangular_region(network,
                                                  'pop_post',
                                                  'SupBasket',
                                                  num_bask,
                                                  0,0,0,
                                                  xDim,yDim,zDim)

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
                                
                                
    total_conns = 0
    if connections:

        proj = oc.add_probabilistic_projection(network,
                                        "proj0",
                                        pop_pre,
                                        pop_post,
                                        syn1.id,
                                        0.3,
                                        weight=0.05,
                                        delay=global_delay)
        if proj:                           
            total_conns += len(proj.connection_wds)
        
        
    if num_rs != 2 or num_bask!=2:
        new_reference = '%s_%scells_%sconns'%(nml_doc.id,num_rs+num_bask,total_conns)
        network.id = new_reference
        nml_doc.id = new_reference

    nml_file_name = '%s.net.%s'%(network.id,'nml.h5' if format == 'hdf5' else 'nml')
    oc.save_network(nml_doc, 
                    nml_file_name, 
                    validate=(format=='xml'),
                    format = format)

    if format=='xml':
        lems_file_name = oc.generate_lems_simulation(nml_doc, network, 
                                nml_file_name, 
                                duration =      duration, 
                                dt =            0.025)
    else:
        lems_file_name = None
                                
    return nml_doc, nml_file_name, lems_file_name


if __name__ == '__main__':
    
    if '-test' in sys.argv:
        
        generate(num_rs = 2,
                 num_bask=0,
                 duration = 50,
                 global_delay = 2)
                 
    if '-large' in sys.argv:
        
        generate(num_rs = 10,
                 num_bask=10,
                 duration = 50,
                 global_delay = 2)
                 
    else:
        generate(global_delay = 5)
