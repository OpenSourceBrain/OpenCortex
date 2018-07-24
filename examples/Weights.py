import opencortex.core as oc
import sys

info = '''
Simple network having multiple connections (chemical/gap junction/analog) with variable weights
'''


def generate(reference = "Weights",
             num_each = 6,
             connections=True,
             duration = 1000,
             format='xml'):

    nml_doc, network = oc.generate_network(reference)

    cell_id = 'HH_477127614'
    cell = oc.include_opencortex_cell(nml_doc, 'AllenInstituteCellTypesDB_HH/%s.cell.nml'%cell_id)

    xDim = 500
    yDim = 500
    zDim = 30

    pop_pre = oc.add_population_in_rectangular_region(network, 'pop_pre',
                                                  cell_id, num_each,
                                                  0,0,0, xDim,yDim,zDim,
                                                  color='.8 0 0')

    pop_post_chem = oc.add_population_in_rectangular_region(network, 'pop_post_chem',
                                                  cell_id, num_each,
                                                  0,yDim,0, xDim,yDim,zDim,
                                                  color='0 0 .8')
                                                  
    pop_post_gap = oc.add_population_in_rectangular_region(network, 'pop_post_gap',
                                                  cell_id, num_each,
                                                  xDim,yDim,0, xDim,yDim,zDim,
                                                  color='0 .8 0')

    ampa_syn = oc.add_exp_two_syn(nml_doc, id="AMPA_syn", 
                             gbase="10nS", erev="0mV",
                             tau_rise="0.003s", tau_decay="0.0031s")

    gj_syn = oc.add_gap_junction_synapse(nml_doc, id="gj0", 
                             conductance="1nS")


    pfs = oc.add_poisson_firing_synapse(nml_doc, id="poissonFiringSyn",
                                       average_rate="20 Hz", synapse_id=ampa_syn.id)

    oc.add_inputs_to_population(network, "Stim0",
                                pop_pre, pfs.id, all_cells=True)


    if connections:
        
        proj_chem = oc.add_probabilistic_projection(network,
                                "proj_chem",
                                pop_pre,
                                pop_post_chem,
                                ampa_syn.id,
                                0.3,
                                weight=1,
                                delay=5)
                                
        for conn in proj_chem.connection_wds:
            if conn.get_pre_cell_id() < 3 and conn.get_post_cell_id() < 3:
                conn.weight = 0.5
            print conn
            
        proj_gap = oc.add_targeted_electrical_projection(nml_doc, 
                                        network,
                                        "proj_gap",
                                        pop_pre,
                                        pop_post_gap,
                                        targeting_mode='convergent',
                                        synapse_list=[gj_syn.id],
                                        pre_segment_group = 'soma_group',
                                        post_segment_group = 'soma_group',
                                        number_conns_per_cell=3)
                          
        for conn in network.electrical_projections[0].electrical_connection_instance_ws:
            conn.weight = conn.get_pre_cell_id() + conn.get_post_cell_id()
            print conn




    nml_file_name = '%s.net.%s'%(network.id,'nml.h5' if format == 'hdf5' else 'nml')
    oc.save_network(nml_doc, 
                    nml_file_name, 
                    validate=(format=='xml'),
                    format = format)

    if format=='xml':

        lems_file_name = oc.generate_lems_simulation(nml_doc, network, 
                                nml_file_name, 
                                duration =      duration, 
                                dt =            0.025,
                                gen_plots_for_all_v = True,
                                gen_saves_for_all_v = True)
    else:
        lems_file_name = None

    return nml_doc, nml_file_name, lems_file_name


if __name__ == '__main__':

    if '-test' in sys.argv:

        generate(num_pre = 1,
                 duration = 1000)

    else:
        generate()
