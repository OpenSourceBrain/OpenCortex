import opencortex.core as oc
import sys

'''
Simple network using electrical connections (gap junctions)

'''


def generate(reference = "GapJunctions",
             num_pre = 5,
             num_post = 2,
             connections=True,
             duration = 1000,
             segments_to_plot_record = {'pop_pre':[0],'pop_post':[0]},
             format='xml'):


    nml_doc, network = oc.generate_network(reference)

    oc.include_opencortex_cell(nml_doc, 'acnet2/pyr_4_sym.cell.nml')

    xDim = 500
    yDim = 50
    zDim = 500

    pop_pre = oc.add_population_in_rectangular_region(network, 'pop_pre',
                                                  'pyr_4_sym', num_pre,
                                                  0,0,0, xDim,yDim,zDim,
                                                  color='0 .8 0')

    pop_post = oc.add_population_in_rectangular_region(network, 'pop_post',
                                                  'pyr_4_sym', num_post,
                                                  0,yDim,0, xDim,yDim+yDim,zDim,
                                                  color='0 .8 0.8')

    ampa_syn = oc.add_exp_two_syn(nml_doc, id="AMPA_syn", 
                             gbase="30e-9S", erev="0mV",
                             tau_rise="0.003s", tau_decay="0.0031s")

    gj_syn = oc.add_gap_junction_synapse(nml_doc, id="gj0", 
                             conductance="5nS")


    pfs = oc.add_poisson_firing_synapse(nml_doc, id="poissonFiringSyn",
                                       average_rate="30 Hz", synapse_id=ampa_syn.id)

    oc.add_inputs_to_population(network, "Stim0",
                                pop_pre, pfs.id, all_cells=True)


    total_conns = 0
    if connections:

        this_syn=gj_syn.id
        proj = oc.add_targeted_electrical_projection(nml_doc, 
                                        network,
                                        "Proj0",
                                        pop_pre,
                                        pop_post,
                                        targeting_mode='convergent',
                                        synapse_list=[this_syn],
                                        pre_segment_group = 'soma_group',
                                        post_segment_group = 'dendrite_group',
                                        number_conns_per_cell=3)
        if proj:                           
            total_conns += len(proj[0].electrical_connections)+len(proj[0].electrical_connection_instances)




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

        generate(num_pre = 1,
                 num_post=1,
                 duration = 1000)

    else:
        generate()
