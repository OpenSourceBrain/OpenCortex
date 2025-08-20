#!/bin/bash
set -ex

pip install .
cd opencortex/test
pytest -vs
cd ../../examples

cleanGenNML.sh
./regenerateAll.sh

time omv all -V
date

echo "Finished OpenCortex tests"
