from neuromllite import (
    Network,
    Cell,
    InputSource,
    Population,
    Synapse,
    RectangularRegion,
    RandomLayout,
)
from neuromllite import Input, Simulation
import sys

################################################################################
###   Build new network

net = Network(id="SimpleNet")
net.notes = "Simple network with single population"
net.temperature = 32.0

cell = Cell(
    id="RS", neuroml2_source_file="../../NeuroML2/prototypes/izhikevich/RS.cell.nml"
)
net.cells.append(cell)

syn = Synapse(
    id="ampa",
    neuroml2_source_file="../../NeuroML2/prototypes/synapses/ampa.synapse.nml",
)
net.synapses.append(syn)

input_source = InputSource(
    id="poissonFiringSyn",
    neuroml2_input="poissonFiringSynapse",
    parameters={"average_rate": "50Hz", "synapse": syn.id, "spike_target": "./ampa"},
)
net.input_sources.append(input_source)

r1 = RectangularRegion(id="region1", x=0, y=0, z=0, width=100, height=100, depth=100)
net.regions.append(r1)

p0 = Population(
    id="RS_pop",
    size=3,
    component=cell.id,
    properties={"color": "0 .8 0"},
    random_layout=RandomLayout(region=r1.id),
)

net.populations.append(p0)

net.inputs.append(
    Input(id="Stim0", input_source=input_source.id, population=p0.id, percentage=100)
)

print(net.to_json())
new_file = net.to_json_file("%s.json" % net.id)


################################################################################
###   Build Simulation object & save as JSON

sim = Simulation(
    id="SimSimpleNet",
    network=new_file,
    duration="1000",
    dt="0.025",
    record_traces={"all": "*"},
)

sim.to_json_file()


################################################################################
###   Run in some simulators

from neuromllite.NetworkGenerator import check_to_generate_or_run

check_to_generate_or_run(sys.argv, sim)
