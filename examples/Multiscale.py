'''
Generates a NeuroML 2 file with many types of cells, populations and inputs
for testing purposes
'''

import opencortex.core as oc
import sys
import numpy as np
import pylab as pl

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
                 
    reference = ("Multiscale__g%s__i%s"%(ratio_inh_exc,input_rate)).replace('.','_')
                    

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
            
            traces, events = oc.simulate_network(lems_file_name,
                     run_in_simulator,
                     max_memory='4000M',
                     nogui=True,
                     load_saved_data=True,
                     reload_events=True,
                     plot=False,
                     verbose=False)
                     
                     
            print("Reloaded traces: %s"%traces.keys())
            #print("Reloaded events: %s"%events.keys())
            
            exc_rate = 0
            inh_rate = 0
            for ek in events.keys():
                rate = 1000 * len(events[ek])/float(duration)
                print("Cell %s has rate %s Hz"%(ek,rate))
                if 'popExc' in ek:
                    exc_rate += rate/num_exc
                if 'popInh' in ek:
                    inh_rate += rate/num_inh
                    
            print("Run %s: Exc rate: %s Hz; Inh rate %s Hz"%(reference,exc_rate, inh_rate))
                     
            return exc_rate, inh_rate
        
    else:
        lems_file_name = None
                                
    return nml_doc, nml_file_name, lems_file_name
                         
                         
                         
def _plot_(X, g_rng, i_rng, sbplt=111, ttl=[]):
    ax = pl.subplot(sbplt)
    pl.title(ttl)
    pl.imshow(X, origin='lower', interpolation='none')
    pl.xlabel('Ratio inh/exc')
    pl.ylabel('Input (Hz)')
    ax.set_xticks(range(0,len(g_rng))); ax.set_xticklabels(g_rng)
    ax.set_yticks(range(0,len(i_rng))); ax.set_yticklabels(i_rng)
    pl.colorbar()


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
        
        duration = 600
        run_in_simulator='jNeuroML_NEURON'
        #run_in_simulator='jNeuroML_NetPyNE'
        scalePops = 1
        
        quick = False
        quick = True
        
        g_rng = np.arange(.5, 4.5, .5)
        i_rng = np.arange(50, 400, 50)
        
        if quick:
            g_rng = [2,3,4]
            i_rng = [100,150,200]
            duration = 200
            scalePops = .5



        Rexc = np.zeros((len(g_rng), len(i_rng)))
        Rinh = np.zeros((len(g_rng), len(i_rng)))
        
        count=1
        for i1, g in enumerate(g_rng):
            for i2, i in enumerate(i_rng):
                print("====================================")
                print(" Run %s of %s: g = %s; i=%s"%(count, len(g_rng)*len(i_rng), g, i))
                info = generate(scalePops = scalePops,
                    scalex=2,
                    scalez=2,
                    duration = duration,
                    max_in_pop_to_plot_and_save = 5,
                    global_delay = 2,
                    ratio_inh_exc = g,
                    input_rate=i,
                    run_in_simulator=run_in_simulator)
                    
                Rexc[i1,i2] = info[0]
                Rinh[i1,i2] = info[1]
                count+=1
                    
                

        fig = pl.figure(figsize=(16,8))
        info = "%s: scale %s, %s ms"%(run_in_simulator,scalePops, duration)

        fig.canvas.set_window_title(info)
        pl.suptitle(info)

        _plot_(Rexc.T, g_rng, i_rng, 221, 'Rates Exc (Hz)')
        _plot_(Rinh.T, g_rng, i_rng, 222, 'Rates Inh (Hz)')
        

        pl.subplots_adjust(wspace=.3, hspace=.3)


        pl.savefig('%s_%s_%sms.png'%(run_in_simulator,scalePops, duration), bbox_inches='tight')
        print("Finished: "+info)
        pl.show()
        

    else:
        generate(gen_spike_saves_for_all_somas = False)