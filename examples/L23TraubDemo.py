
import opencortex.build as oc

import sys

DEFAULT_RS_POP_SIZE = 1
DEFAULT_BASK_POP_SIZE = 1

def generate(reference = "L23TraubDemo",
             num_rs = DEFAULT_RS_POP_SIZE,
             num_bask = DEFAULT_BASK_POP_SIZE,
             scalex=1,
             scaley=1,
             scalez=1,
             connections=False,
             poisson_inputs=True,
             offset_curents=False,
             global_delay = 0,
             duration = 300,
             segments_to_plot_record = {'pop_rs':[0],'pop_bask':[0]},
             format='xml'):


    nml_doc, network = oc.generate_network(reference)

    #oc.add_cell_and_channels(nml_doc, 'acnet2/pyr_4_sym.cell.nml','pyr_4_sym')
    oc.add_cell_and_channels(nml_doc, 'Thalamocortical/L23PyrRS.cell.nml','L23PyrRS')
    oc.add_cell_and_channels(nml_doc, 'Thalamocortical/SupBasket.cell.nml','SupBasket')
    
    xDim = 500*scalex
    yDim = 200*scaley
    zDim = 500*scalez

    pop_rs = oc.add_population_in_rectangular_region(network,
                                                  'pop_rs',
                                                  'L23PyrRS',
                                                  num_rs,
                                                  0,0,0,
                                                  xDim,yDim,zDim)

    pop_bask = oc.add_population_in_rectangular_region(network,
                                                  'pop_bask',
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
                             
                            
    if poisson_inputs:

        pfs = oc.add_poisson_firing_synapse(nml_doc,
                                           id="poissonFiringSyn",
                                           average_rate="150 Hz",
                                           synapse_id=syn0.id)

        oc.add_inputs_to_population(network,
                                    "Stim0",
                                    pop_rs,
                                    pfs.id,
                                    all_cells=True)
                                    
        oc.add_inputs_to_population(network,
                                    "Stim1",
                                    pop_bask,
                                    pfs.id,
                                    all_cells=True)
    if offset_curents:

        pg0 = oc.add_pulse_generator(nml_doc,
                               id="pg0",
                               delay="0ms",
                               duration="%sms"%duration,
                               amplitude="0.5nA")

        oc.add_inputs_to_population(network,
                                    "Stim0",
                                    pop_rs,
                                    pg0.id,
                                    all_cells=True)

        oc.add_inputs_to_population(network,
                                    "Stim1",
                                    pop_bask,
                                    pg0.id,
                                    all_cells=True)
                                
                                
    total_conns = 0
    if connections:

        proj = oc.add_probabilistic_projection(network,
                                        "proj0",
                                        pop_rs,
                                        pop_bask,
                                        syn1.id,
                                        0.3,
                                        weight=0.05,
                                        delay=global_delay)
        if proj:                           
            total_conns += len(proj.connection_wds)
        
        
    if num_rs != DEFAULT_RS_POP_SIZE or num_bask!=DEFAULT_BASK_POP_SIZE:
        new_reference = '%s_%scells_%sconns'%(nml_doc.id,num_rs+num_bask,total_conns)
        network.id = new_reference
        nml_doc.id = new_reference

    nml_file_name = '%s.net.%s'%(network.id,'nml.h5' if format == 'hdf5' else 'nml')
    oc.save_network(nml_doc, 
                    nml_file_name, 
                    validate=(format=='xml'),
                    format = format)

    if format=='xml':
        gen_plots_for_quantities = {}   #  Dict with displays vs lists of quantity paths
        gen_saves_for_quantities = {}   #  Dict with file names vs lists of quantity paths
        
        for pop in segments_to_plot_record.keys():
            pop_nml = network.get_by_id(pop)
            if pop_nml is not None and pop_nml.size>0:
                for i in range(int(pop_nml.size)):
                    gen_plots_for_quantities['Display_%s_%i_v'%(pop,i)] = []
                    gen_saves_for_quantities['Sim_%s.%s.%i.v.dat'%(nml_doc.id,pop,i)] = []

                    for seg in segments_to_plot_record[pop]:
                        quantity = '%s/%i/%s/%i/v'%(pop,i,pop_nml.component,seg)
                        gen_plots_for_quantities['Display_%s_%i_v'%(pop,i)].append(quantity)
                        gen_saves_for_quantities['Sim_%s.%s.%i.v.dat'%(nml_doc.id,pop,i)].append(quantity)


            
        lems_file_name = oc.generate_lems_simulation(nml_doc, network, 
                                nml_file_name, 
                                duration =      duration, 
                                dt =            0.025,
                                gen_plots_for_all_v = False,
                                gen_plots_for_quantities = gen_plots_for_quantities,
                                gen_saves_for_all_v = False,
                                gen_saves_for_quantities = gen_saves_for_quantities)
    else:
        lems_file_name = None
                                
    return nml_doc, nml_file_name, lems_file_name


if __name__ == '__main__':
    
    if '-test' in sys.argv:
        
        generate(num_rs = 2,
                 num_bask=0,
                 duration = 50,
                 global_delay = 2)
                 
        generate(num_rs = 0,
                 num_bask=1,
                 duration = 30,
                 poisson_inputs=False,
                 offset_curents=True,
                 segments_to_plot_record = {'pop_bask':[0,117,104,55]})
                 
                 
        
        generate(num_rs = 8,
                 num_bask=4,
                 duration = 200,
                 connections = True,
                 global_delay = 2)
        
        generate(num_rs = 32,
                 num_bask=16,
                 duration = 200,
                 connections = True,
                 global_delay = 2)
    else:
        generate(global_delay = 5)
