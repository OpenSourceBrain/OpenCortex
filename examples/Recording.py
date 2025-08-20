"""
Generates a NeuroML 2 file with a LEMS file recording many details of the network
"""

import opencortex.core as oc


nml_doc, network = oc.generate_network("Recording")


#####   Cells

oc.include_opencortex_cell(nml_doc, "acnet2/pyr_4_sym.cell.nml")

xDim = 500
yDim = 100
zDim = 500
offset = 0


#####   Synapses

synAmpa1 = oc.add_exp_two_syn(
    nml_doc, id="synAmpa1", gbase="1nS", erev="0mV", tau_rise="0.5ms", tau_decay="10ms"
)

synAmpa2 = oc.add_exp_two_syn(
    nml_doc, id="synAmpa2", gbase="0.5nS", erev="0mV", tau_rise="0.5ms", tau_decay="5ms"
)

#####   Input types

pg0 = oc.add_pulse_generator(
    nml_doc, id="pg0", delay="10ms", duration="300ms", amplitude="0.3nA"
)

pg1 = oc.add_pulse_generator(
    nml_doc, id="pg1", delay="50ms", duration="400ms", amplitude="0.35nA"
)

pfs = oc.add_poisson_firing_synapse(
    nml_doc, id="poissonFiringSyn", average_rate="150 Hz", synapse_id=synAmpa2.id
)

#####   Populations


pop0 = oc.add_population_in_rectangular_region(
    network, "pop0", "pyr_4_sym", 5, 0, offset, 0, xDim, yDim, zDim, color="0 .8 0"
)

offset += yDim

#####   Projections


oc.add_probabilistic_projection(network, "proj0", pop0, pop0, synAmpa1.id, 0.5)

#####   Inputs

oc.add_inputs_to_population(network, "Stim0", pop0, pg0.id, only_cells=[0])

oc.add_inputs_to_population(network, "Stim1", pop0, pg1.id, only_cells=[1])


oc.add_inputs_to_population(network, "Stim2", pop0, pfs.id, all_cells=True)


nml_file_name = "%s.net.nml" % network.id
oc.save_network(nml_doc, nml_file_name, validate=True)

oc.generate_lems_simulation(
    nml_doc,
    network,
    nml_file_name,
    duration=500,
    dt=0.005,
    plot_all_segments=True,
    save_all_segments=True,
)
