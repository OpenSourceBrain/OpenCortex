from setuptools import setup

import opencortex
version = opencortex.__version__

setup(
    name='OpenCortex',
    version=version,
    author='Rokas Stanislovas and Padraig Gleeson',
    author_email='p.gleeson@gmail.com',
    packages = ['opencortex',
                'opencortex.core',
                'opencortex.build',
                'opencortex.test',
                'opencortex.utils'],
    package_data={
        'opencortex': [
            '../NeuroML2/prototypes/iaf/*',
            '../NeuroML2/prototypes/izhikevich/*',
            '../NeuroML2/prototypes/Thalamocortical/*',
            '../NeuroML2/prototypes/BlueBrainProject_NMC/*',
            '../NeuroML2/prototypes/AllenInstituteCellTypesDB_HH/*',
            '../NeuroML2/prototypes/L23Pyr_SmithEtAl2013/*',
            '../NeuroML2/prototypes/acnet2/*']}, 

    url='https://github.com/OpenSourceBrain/OpenCortex',
    license='LICENSE.lesser',
    description='A framework for building cortical network models',
    long_description=open('README.md').read(),
    install_requires=[
        'libNeuroML>=0.2.39',
        'pylems',
        'pyNeuroML>=0.3.11',
        'matplotlib'],
    dependency_links=[
      'git+https://github.com/NeuralEnsemble/libNeuroML.git@development#egg=libNeuroML-0.2.10'
    ],
    classifiers = [
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.2',
        'Topic :: Scientific/Engineering']
)
