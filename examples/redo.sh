#!/bin/bash
set -ex
cd ..
python setup.py install
cd opencortex/test
nosetests -vs
cd ../../examples

cleanGenNML.sh
./regenerateAll.sh

time omv all -V
date

echo "Finished OpenCortex tests"
