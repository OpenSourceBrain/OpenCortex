import opencortex.core as oc
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
             global_delay = 0,
             duration = 250,
             segments_to_plot_record = {'pop_pyr':[0],'pop_bask':[0]},
             format='xml'):


    nml_doc, network = oc.generate_network(reference)

    oc.include_opencortex_cell(nml_doc, 'acnet2/pyr_4_sym.cell.nml')
    oc.include_opencortex_cell(nml_doc, 'acnet2/bask.cell.nml')
    
    xDim = 500*scalex
    yDim = 50*scaley
    zDim = 500*scalez

    pop_pyr = oc.add_population_in_rectangular_region(network, 'pop_pyr',
                                                  'pyr_4_sym', num_pyr,
                                                  0,0,0, xDim,yDim,zDim,
                                                  color='.8 0 0')

    pop_bask = oc.add_population_in_rectangular_region(network, 'pop_bask',
                                                  'bask', num_bask,
                                                  0,yDim,0, xDim,yDim+yDim,zDim,
                                                  color='0 0 .8')
                   
    ampa_syn = oc.add_exp_two_syn(nml_doc, id="AMPA_syn", 
                             gbase="30e-9S", erev="0mV",
                             tau_rise="0.003s", tau_decay="0.0031s")

    ampa_syn_inh = oc.add_exp_two_syn(nml_doc, id="AMPA_syn_inh", 
                             gbase="0.15e-9S", erev="0mV",
                             tau_rise="0.003s", tau_decay="0.0031s")

    gaba_syn = oc.add_exp_two_syn(nml_doc, id="GABA_syn", 
                             gbase="0.6e-9S", erev="-0.080V",
                             tau_rise="0.005s", tau_decay="0.012s")

    gaba_syn_inh = oc.add_exp_two_syn(nml_doc, id="GABA_syn_inh", 
                             gbase="0S", erev="-0.080V",
                             tau_rise="0.003s", tau_decay="0.008s")

    pfs = oc.add_poisson_firing_synapse(nml_doc, id="poissonFiringSyn",
                                       average_rate="30 Hz", synapse_id=ampa_syn.id)
                                
                                
    total_conns = 0
    if connections:

        this_syn=ampa_syn.id
        proj = oc.add_targeted_projection(network,
                                        "Proj_pyr_pyr",
                                        pop_pyr,
                                        pop_pyr,
                                        targeting_mode='convergent',
                                        synapse_list=[this_syn],
                                        pre_segment_group = 'soma_group',
                                        post_segment_group = 'dendrite_group',
                                        number_conns_per_cell=7,
                                        delays_dict = {this_syn:global_delay})
        if proj:                           
            total_conns += len(proj[0].connection_wds)

        this_syn=ampa_syn_inh.id
        proj = oc.add_targeted_projection(network,
                                        "Proj_pyr_bask",
                                        pop_pyr,
                                        pop_bask,
                                        targeting_mode='convergent',
                                        synapse_list=[this_syn],
                                        pre_segment_group = 'soma_group',
                                        post_segment_group = 'all',
                                        number_conns_per_cell=21,
                                        delays_dict = {this_syn:global_delay})
        if proj:                           
            total_conns += len(proj[0].connection_wds)

        this_syn=gaba_syn.id
        proj = oc.add_targeted_projection(network,
                                        "Proj_bask_pyr",
                                        pop_bask,
                                        pop_pyr,
                                        targeting_mode='convergent',
                                        synapse_list=[this_syn],
                                        pre_segment_group = 'soma_group',
                                        post_segment_group = 'all',
                                        number_conns_per_cell=21,
                                        delays_dict = {this_syn:global_delay})
        if proj:                           
            total_conns += len(proj[0].connection_wds)

        this_syn=gaba_syn_inh.id
        proj = oc.add_targeted_projection(network,
                                        "Proj_bask_bask",
                                        pop_bask,
                                        pop_bask,
                                        targeting_mode='convergent',
                                        synapse_list=[this_syn],
                                        pre_segment_group = 'soma_group',
                                        post_segment_group = 'all',
                                        number_conns_per_cell=5,
                                        delays_dict = {this_syn:global_delay})
        if proj:                           
            total_conns += len(proj[0].connection_wds)
            
            
    oc.add_targeted_inputs_to_population(network, "Stim0",
                                pop_pyr, pfs.id, 
                                segment_group='soma_group',
                                number_per_cell = 1,
                                all_cells=True)
        
        
    if num_pyr != 48 or num_bask!=12:
        new_reference = '%s_%scells_%sconns'%(nml_doc.id,num_pyr+num_bask,total_conns)
        network.id = new_reference
        nml_doc.id = new_reference
        
    nml_file_name = '%s.net.%s'%(network.id,'nml.h5' if format == 'hdf5' else 'nml')
    target_dir = 'HDF5/' if format == 'hdf5' else './'
    
    oc.save_network(nml_doc, 
                    nml_file_name, 
                    validate=(format=='xml'),
                    format = format,
                    target_dir=target_dir)


    gen_plots_for_quantities = {}   #  Dict with displays vs lists of quantity paths
    gen_saves_for_quantities = {}   #  Dict with file names vs lists of quantity paths

    for pop in segments_to_plot_record.keys():
        pop_nml = network.get_by_id(pop)
        if pop_nml is not None and pop_nml.size>0:

            group = len(segments_to_plot_record[pop]) == 1
            if group:
                display = 'Display_%s_v'%(pop)
                file_ = 'Sim_%s.%s.v.dat'%(nml_doc.id,pop)
                gen_plots_for_quantities[display] = []
                gen_saves_for_quantities[file_] = []

            for i in range(int(pop_nml.size)):
                if not group:
                    display = 'Display_%s_%i_v'%(pop,i)
                    file_ = 'Sim_%s.%s.%i.v.dat'%(nml_doc.id,pop,i)
                    gen_plots_for_quantities[display] = []
                    gen_saves_for_quantities[file_] = []

                for seg in segments_to_plot_record[pop]:
                    quantity = '%s/%i/%s/%i/v'%(pop,i,pop_nml.component,seg)
                    gen_plots_for_quantities[display].append(quantity)
                    gen_saves_for_quantities[file_].append(quantity)

    lems_file_name = oc.generate_lems_simulation(nml_doc, 
                            network, 
                            target_dir+nml_file_name, 
                            duration =      duration, 
                            dt =            0.025,
                            gen_plots_for_all_v = False,
                            gen_plots_for_quantities = gen_plots_for_quantities,
                            gen_saves_for_all_v = False,
                            gen_saves_for_quantities = gen_saves_for_quantities,
                            target_dir=target_dir)
                                
    return nml_doc, nml_file_name, lems_file_name


if __name__ == '__main__':
    
    if '-test' in sys.argv:
        
        generate(num_pyr = 2,
                 num_bask=2,
                 duration = 500,
                 global_delay = 2)
                 
        generate(num_pyr = 1,
                 num_bask=0,
                 duration = 500,
                 segments_to_plot_record = {'pop_pyr':range(9)})
    else:
        generate(global_delay = 1)
        generate(global_delay = 1, format='hdf5')
