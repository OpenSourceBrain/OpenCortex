
import neuroml
from pyneuroml import pynml
from pyneuroml.lems.LEMSSimulation import LEMSSimulation

import random
random.seed(12345)

ref = "SimpleNet"
nml_doc = neuroml.NeuroMLDocument(id=ref)

# Define izh cell
izh0 = neuroml.Izhikevich2007Cell(id="RS", v0 = "-60mV", C="100 pF", k = "3 nS_per_mV", vr = "-60 mV", vt = "-50 mV", vpeak = "50 mV", a = "0.01 per_ms", b = "5 nS", c = "-60 mV", d = "400 pA")
                        
nml_doc.izhikevich2007_cells.append(izh0)

# Define synapse
syn0 = neuroml.ExpTwoSynapse(id="syn0", gbase="2nS",
                             erev="0mV",
                             tau_rise="0.5ms",
                             tau_decay="10ms")
nml_doc.exp_two_synapses.append(syn0)

#<poissonFiringSynapse id="poissonFiringSyn" averageRate="50 Hz" synapse="synInput" spikeTarget="./synInput"/>
pfs = neuroml.PoissonFiringSynapse(id="poissonFiringSyn",
                                   average_rate="50 Hz",
                                   synapse=syn0.id, 
                                   spike_target="./%s"%syn0.id)
nml_doc.poisson_firing_synapses.append(pfs)

# Create network
net = neuroml.Network(id=ref+"_network")
nml_doc.networks.append(net)


# Create populations
size0 = 5
pop0 = neuroml.Population(id="Pop0", size = size0,
                          component=izh0.id)
net.populations.append(pop0)



for i in range(size0):
    expInp = neuroml.ExplicitInput(target='%s[%i]'%(pop0.id,i),
                                   input=pfs.id,
                                   destination="synapses")
    net.explicit_inputs.append(expInp)


import neuroml.writers as writers

nml_file = '%s.nml'%ref
writers.NeuroMLWriter.write(nml_doc, nml_file)


print("Written network file to: "+nml_file)


###### Validate the NeuroML ######    

from neuroml.utils import validate_neuroml2

validate_neuroml2(nml_file)

# Create a LEMSSimulation to manage creation of LEMS file
duration = 1000  # ms
dt = 0.01  # ms
ls = LEMSSimulation(ref, duration, dt)


# Point to network as target of simulation
ls.assign_simulation_target(net.id)

# Include generated/existing NeuroML2 files
ls.include_neuroml2_file(nml_file)

# Specify Displays and Output Files
disp0 = "display_voltages0"
ls.create_display(disp0, "Voltages Pop0", "-68", "-47")

of0 = 'Volts0_file'
ls.create_output_file(of0, "v_pop0.dat")

max_traces = 10
max_saves = 20

for i in range(size0):
    quantity = "%s[%i]/v"%(pop0.id, i)
    if i<max_traces:
        ls.add_line_to_display(disp0, "%s[%i]: Vm"%(pop0.id,i), quantity, "1mV", pynml.get_next_hex_color())
    if i<max_saves:
        ls.add_column_to_output_file(of0, 'v%i'%i, quantity)
    

# Save to LEMS XML file
lems_file_name = ls.save_to_file()

# Run with jNeuroML
#results1 = pynml.run_lems_with_jneuroml(lems_file_name, nogui=True, load_saved_data=True, plot=True)

# Run with jNeuroML_NEURON
#results1 = pynml.run_lems_with_jneuroml_neuron(lems_file_name, nogui=True, load_saved_data=True, plot=True)
