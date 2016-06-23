from Balanced import generate


nml_doc, nml_file_name, lems_file_name = generate(num_bbp =10,
     scalePops = 20,
     scalex=2,
     scalez=2,
     connections=True,
     format='hdf5')
     
from neuroml.loaders import NeuroMLHdf5Loader

nml_doc2 = NeuroMLHdf5Loader.load(nml_file_name)


for doc in [nml_doc,nml_doc2]:
    doc.summary()
    