'''
Generates a NeuroML 2 file with many types of cells, populations and inputs
for testing purposes
'''

import opencortex.core as oc
import sys
import numpy as np

min_pop_size = 3

def scale_pop_size(baseline, scale):
    return max(min_pop_size, int(baseline*scale))


def generate(scalePops = 1,
             scalex=1,
             scaley=1,
             scalez=1,
             ratio_inh_exc=2,
             connections=True,
             duration = 1000,
             input_rate = 150,
             global_delay = 0,
             max_in_pop_to_plot_and_save = 5,
             gen_spike_saves_for_all_somas = True,
             format='xml',
             run_in_simulator=None):
                 
    reference = "Multiscale_g%s_i%s"%(ratio_inh_exc,input_rate)
                    

    num_exc = scale_pop_size(80,scalePops)
    num_inh = scale_pop_size(40,scalePops)
    
    nml_doc, network = oc.generate_network(reference)

    oc.include_opencortex_cell(nml_doc, 'AllenInstituteCellTypesDB_HH/HH_477127614.cell.nml')
    oc.include_opencortex_cell(nml_doc, 'AllenInstituteCellTypesDB_HH/HH_476686112.cell.nml')
    

    xDim = 400*scalex
    yDim = 500*scaley
    zDim = 300*scalez

    xs = -200
    ys = -150
    zs = 100

    #####   Synapses
    
    exc_syn_nS = 1.

    synAmpa1 = oc.add_exp_two_syn(nml_doc, id="synAmpa1", gbase="%snS"%exc_syn_nS,
                             erev="0mV", tau_rise="0.5ms", tau_decay="5ms")

    synGaba1 = oc.add_exp_two_syn(nml_doc, id="synGaba1", gbase="%snS"%(exc_syn_nS*ratio_inh_exc),
                             erev="-80mV", tau_rise="1ms", tau_decay="20ms")

    #####   Input types


    pfs1 = oc.add_poisson_firing_synapse(nml_doc,
                                       id="psf1",
                                       average_rate="%s Hz"%input_rate,
                                       synapse_id=synAmpa1.id)


    #####   Populations

    popExc = oc.add_population_in_rectangular_region(network,
                                                  'popExc',
                                                  'HH_477127614',
                                                  num_exc,
                                                  xs,ys,zs,
                                                  xDim,yDim,zDim)

    popInh = oc.add_population_in_rectangular_region(network,
                                                  'popInh',
                                                  'HH_476686112',
                                                  num_inh,
                                                  xs,ys,zs,
                                                  xDim,yDim,zDim)


    #####   Projections

    total_conns = 0
    if connections:
        
        proj = oc.add_probabilistic_projection(network, "proj0",
                                        popExc, popExc,
                                        synAmpa1.id, 0.5, delay = global_delay)
        total_conns += len(proj.connection_wds)

        proj = oc.add_probabilistic_projection(network, "proj1",
                                        popExc, popInh,
                                        synAmpa1.id, 0.7, delay = global_delay)
        total_conns += len(proj.connection_wds)

        proj = oc.add_probabilistic_projection(network, "proj3",
                                        popInh, popExc,
                                        synGaba1.id, 0.7, delay = global_delay)
        total_conns += len(proj.connection_wds)

        proj = oc.add_probabilistic_projection(network, "proj4",
                                        popInh, popInh,
                                        synGaba1.id, 0.5, delay = global_delay)
        total_conns += len(proj.connection_wds)

                                        
        total_conns += len(proj.connection_wds)

    #####   Inputs

    oc.add_inputs_to_population(network, "Stim0",
                                popExc, pfs1.id,
                                all_cells=True)



    #####   Save NeuroML and LEMS Simulation files      
    

    nml_file_name = '%s.net.%s'%(network.id,'nml.h5' if format == 'hdf5' else 'nml')
    oc.save_network(nml_doc, 
                    nml_file_name, 
                    validate=(format=='xml'),
                    format = format)

    if format=='xml':
        
        plot_v = {popExc.id:[],popInh.id:[]}
        exc_traces = '%s_%s_v.dat'%(network.id,popExc.id)
        inh_traces = '%s_%s_v.dat'%(network.id,popInh.id)
        save_v = {exc_traces:[], inh_traces:[]}
        
        
        for i in range(min(max_in_pop_to_plot_and_save,num_exc)):
            plot_v[popExc.id].append("%s/%i/%s/v"%(popExc.id,i,popExc.component))
            save_v[exc_traces].append("%s/%i/%s/v"%(popExc.id,i,popExc.component))
            
        for i in range(min(max_in_pop_to_plot_and_save,num_inh)):
            plot_v[popInh.id].append("%s/%i/%s/v"%(popInh.id,i,popInh.component))
            save_v[inh_traces].append("%s/%i/%s/v"%(popInh.id,i,popInh.component))
            
        lems_file_name = oc.generate_lems_simulation(nml_doc, network, 
                                nml_file_name, 
                                duration =      duration, 
                                dt =            0.025,
                                gen_plots_for_all_v = False,
                                gen_plots_for_quantities = plot_v,
                                gen_saves_for_all_v = False,
                                gen_saves_for_quantities = save_v,
                                gen_spike_saves_for_all_somas = gen_spike_saves_for_all_somas)
                                
        
        if run_in_simulator:
            
            results = oc.simulate_network(lems_file_name,
                     run_in_simulator,
                     max_memory='4000M',
                     nogui=True,
                     load_saved_data=True,
                     plot=False,
                     verbose=True)
                     
            return nml_doc, nml_file_name, lems_file_name, results
        
    else:
        lems_file_name = None
                                
    return nml_doc, nml_file_name, lems_file_name
                               

if __name__ == '__main__':
    
    if '-all' in sys.argv:
        generate()
        
        generate(scalePops = 5,
             scalex=2,
             scalez=2,
             connections=False)
        
        generate(scalePops = 5,
             scalex=2,
             scalez=2)
        
        generate(scalePops = 2,
             scalex=2,
             scalez=2,
             duration = 2000)
             
    elif '-test' in sys.argv:
        
        generate(scalePops = 0.2,
             scalex=2,
             scalez=2,
             duration = 1000,
             max_in_pop_to_plot_and_save = 5,
             global_delay = 2,
             input_rate=250)
             
    elif '-paramSweep' in sys.argv:     
        
        g_rng = np.arange(1, 4, .5)
        g_rng = [4]
        
        for g in g_rng:
            nml_doc, nml_file_name, lems_file_name, results = generate(scalePops = 1,
                scalex=2,
                scalez=2,
                duration = 100,
                max_in_pop_to_plot_and_save = 5,
                global_delay = 2,
                ratio_inh_exc = g,
                input_rate=250,
                run_in_simulator='jNeuroML_NEURON')
             
            print("Reloaded: %s"%results.keys())
        

    else:
        generate(gen_spike_saves_for_all_somas = False)