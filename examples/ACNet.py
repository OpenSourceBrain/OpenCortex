import opencortex.build as oc
import sys

'''
Simple network using cells from ACNet model

'''


def generate(reference = "ACNet",
             num_pyr = 48,
             num_bask = 12,
             scalex=1,
             scaley=1,
             scalez=1,
             connections=True,
             global_conn_probability = 0.3,
             global_delay = 0,
             duration = 1000,
             format='xml'):


    nml_doc, network = oc.generate_network(reference)

    oc.add_cell_and_channels(nml_doc, 'acnet2/pyr_4_sym.cell.nml','pyr_4_sym')
    oc.add_cell_and_channels(nml_doc, 'acnet2/bask.cell.nml','bask')
    
    xDim = 500*scalex
    yDim = 200*scaley
    zDim = 500*scalez

    pop_pyr = oc.add_population_in_rectangular_region(network,
                                                  'pop_pyr',
                                                  'pyr_4_sym',
                                                  num_pyr,
                                                  0,0,0,
                                                  xDim,yDim,zDim)

    pop_bask = oc.add_population_in_rectangular_region(network,
                                                  'pop_bask',
                                                  'bask',
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
                                pop_pyr,
                                pfs.id,
                                all_cells=True)
                                
                                
    total_conns = 0
    if connections:

        proj = oc.add_probabilistic_projection(network,
                                        "proj0",
                                        pop_pyr,
                                        pop_bask,
                                        syn1.id,
                                        global_conn_probability,
                                        weight=1,
                                        delay=global_delay)
        if proj:                           
            total_conns += len(proj.connection_wds)
        
        
    if num_pyr != 48 or num_bask!=12:
        new_reference = '%s_%scells_%sconns'%(nml_doc.id,num_pyr+num_bask,total_conns)
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
        
        generate(num_pyr = 2,
                 num_bask=2,
                 duration = 500,
                 global_delay = 2,
                 global_conn_probability=1)
    else:
        generate(global_delay = 1)
