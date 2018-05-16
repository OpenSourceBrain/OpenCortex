'''
Generates a NeuroML 2 network with a number of cells with voltage clamps
'''

import opencortex.core as oc

import sys


def generate(reference = "VClamp",
             poisson_inputs=True,
             use_vclamp=False,
             duration = 500,
             format='xml'):


    nml_doc, network = oc.generate_network(reference)

    oc.include_opencortex_cell(nml_doc, 'Thalamocortical/L23PyrRS.cell.nml')
    
    num_cells = 4

    pop_rs = oc.add_population_in_rectangular_region(network,
                                     'popRS',
                                     'L23PyrRS',
                                     num_cells,
                                     0,0,0,
                                     1000,20,20)

    syn0 = oc.add_exp_two_syn(nml_doc, 
                             id="syn0", 
                             gbase="1nS",
                             erev="0mV",
                             tau_rise="0.5ms",
                             tau_decay="10ms")
                             
                            
    if poisson_inputs:

        pfs = oc.add_transient_poisson_firing_synapse(nml_doc,
                                           id="poissonFiringSyn",
                                           average_rate="20 Hz",
                                           delay="50 ms",
                                           duration="400 ms",
                                           synapse_id=syn0.id)

        oc.add_targeted_inputs_to_population(network, 
                                         "pfs_noise",
                                         pop_rs, 
                                         pfs.id,
                                         segment_group='dendrite_group',
                                         number_per_cell = 100,
                                         all_cells=True)

    all_vclamp_segs = [0, 142, 87]
    vclamp_segs = {0:[],1:[0],2:[0, 142], 3:all_vclamp_segs}
    
    gen_plots_for_quantities = {}   #  Dict with displays vs lists of quantity paths
    gen_saves_for_quantities = {}   #  Dict with file names vs lists of quantity paths
    
    if use_vclamp:
    
        v_clamped = '-70mV'
        
        for cell_id in vclamp_segs:


            for seg_id in vclamp_segs[cell_id]:

                vc = oc.add_voltage_clamp_triple(nml_doc, id='vclamp_cell%i_seg%i'%(cell_id,seg_id), 
                                     delay='0ms', 
                                     duration='%sms'%duration, 
                                     conditioning_voltage=v_clamped,
                                     testing_voltage=v_clamped,
                                     return_voltage=v_clamped, 
                                     simple_series_resistance="1e1ohm",
                                     active = "1")

                vc_dat_file = 'v_clamps_i_cell%s_seg%s.dat'%(cell_id,seg_id)

                gen_saves_for_quantities[vc_dat_file] = []

                oc.add_inputs_to_population(network, "input_vClamp_cell%i_seg%i"%(cell_id,seg_id),
                                            pop_rs, vc.id,
                                            all_cells=False,
                                            only_cells=[cell_id],
                                            segment_ids=[seg_id])  

                q = '%s/%s/%s/%s/%s/i'%(pop_rs.id, cell_id,pop_rs.component,seg_id,vc.id)

                gen_saves_for_quantities[vc_dat_file].append(q)
    

    nml_file_name = '%s.net.%s'%(network.id,'nml.h5' if format == 'hdf5' else 'nml')
    oc.save_network(nml_doc, 
                    nml_file_name, 
                    validate=(format=='xml'),
                    format = format)

    
    segments_to_plot_record = {pop_rs.id:all_vclamp_segs+[20,50,99,139]}
    
    if format=='xml':
        
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
        
        generate()
                 
    else:
        
        generate(use_vclamp=True)
