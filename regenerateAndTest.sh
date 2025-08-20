#!/bin/bash
set -ex

ruff format *.py */*.py */*/*.py */*/*/*.py */*/*/*/*.py

pip install .
cd opencortex/test
pytest -vs
cd ../../examples

cleanGenNML.sh
./regenerateAll.sh

time omv all -V
date

echo "Finished OpenCortex tests"
