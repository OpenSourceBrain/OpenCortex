import opencortex.core as oc
import sys

from neuroml import GradedSynapse
from neuroml import SilentSynapse
from neuroml import ContinuousProjection
from neuroml import ContinuousConnectionInstanceW

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

    pop_post_chem_exc = oc.add_population_in_rectangular_region(network, 'pop_post_chem_exc',
                                                  cell_id, num_each+1,
                                                  0,yDim,0, xDim,yDim,zDim,
                                                  color='0 0 .8')

    pop_post_chem_inh = oc.add_population_in_rectangular_region(network, 'pop_post_chem_inh',
                                                  cell_id, num_each+2,
                                                  xDim,yDim,0, xDim,yDim,zDim,
                                                  color='0 .8 .8')
                                                  
    pop_post_cont = oc.add_population_in_rectangular_region(network, 'pop_post_cont',
                                                  cell_id, num_each+3,
                                                  xDim,0,0, xDim,yDim,zDim,
                                                  color='0 .8 0')

    ampa_syn = oc.add_exp_two_syn(nml_doc, id="AMPA_syn", 
                             gbase="10nS", erev="0mV",
                             tau_rise="2ms", tau_decay="10ms")
                             
    gaba_syn = oc.add_exp_two_syn(nml_doc, id="GABA_syn", 
                             gbase="10nS", erev="-80mV",
                             tau_rise="3ms", tau_decay="30ms")

    gj_syn = oc.add_gap_junction_synapse(nml_doc, id="gj0", 
                             conductance=".05nS")
                             
    
    analog_syn = GradedSynapse(id='analog_syn',
                             conductance="10nS",
                             delta="5mV",
                             Vth="-35mV",
                             k="0.025per_ms",
                             erev="0mV")
    silent_syn = SilentSynapse(id="silent1")
    
    nml_doc.graded_synapses.append(analog_syn)
    nml_doc.silent_synapses.append(silent_syn)


    pfs = oc.add_poisson_firing_synapse(nml_doc, id="poissonFiringSyn",
                                       average_rate="10 Hz", synapse_id=ampa_syn.id)

    oc.add_inputs_to_population(network, "Stim0",
                                pop_pre, pfs.id, all_cells=True)


    if connections:
        
        proj_chem_exc = oc.add_probabilistic_projection(network,
                                "proj_chem_exc",
                                pop_pre,
                                pop_post_chem_exc,
                                ampa_syn.id,
                                0.7,
                                weight=1,
                                delay=5)
                                
        for conn in proj_chem_exc.connection_wds:
            if conn.get_pre_cell_id() < 3 and conn.get_post_cell_id() < 3:
                conn.weight = 0.5
            print conn
        
        proj_chem_inh = oc.add_probabilistic_projection(network,
                                "proj_chem_inh",
                                pop_pre,
                                pop_post_chem_inh,
                                gaba_syn.id,
                                0.7,
                                weight=1,
                                delay=5)
                                
        for conn in proj_chem_inh.connection_wds:
            if conn.get_pre_cell_id() < 3 and conn.get_post_cell_id() < 3:
                conn.weight = 2
            print conn
            
        
        proj_cont = ContinuousProjection(id='proj_cont', \
                           presynaptic_population=pop_pre.id,
                           postsynaptic_population=pop_post_cont.id)
        network.continuous_projections.append(proj_cont)
        
        for i in range(pop_pre.get_size()):
            for j in range(pop_post_cont.get_size()):
                conn0 = ContinuousConnectionInstanceW(id='%s'%(j+i*pop_pre.get_size()), \
                           pre_cell='../%s/%s/%s'%(pop_pre.id,i,cell_id),
                           post_cell='../%s/%s/%s'%(pop_post_cont.id,j,cell_id),
                           pre_component=silent_syn.id,
                           post_component=analog_syn.id,
                           weight=(i+j)/10.0)
                proj_cont.continuous_connection_instance_ws.append(conn0)
        
        
            
        gj_pops = [pop_pre, pop_post_chem_exc, pop_post_chem_inh, pop_post_cont]
        
        for pre in gj_pops:
            for post in gj_pops:
                
                proj_gap = oc.add_targeted_electrical_projection(nml_doc, 
                                                network,
                                                "proj_",
                                                pre,
                                                post,
                                                targeting_mode='convergent',
                                                synapse_list=[gj_syn.id],
                                                pre_segment_group = 'soma_group',
                                                post_segment_group = 'soma_group',
                                                number_conns_per_cell=3)

                for conn in network.electrical_projections[-1].electrical_connection_instance_ws:
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
