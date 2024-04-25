#!/bin/sh
set -euxo pipefail

host_code=$1
if [ $# -eq 2 ]; then
    cu_code=$1
    kernel=$2
elif [ $# -eq 3 ]; then
    cu_code=$2
    kernel=$3
else
    echo "Usage: ./dynamic_transform.sh <host code file (leave out if same as kernel)> <kernel file> <kernel name>"
    exit 1
fi

g++ -std=c++0x src/trace.cpp -c -l
g++ -std=c++0x src/Dynamic_Analysis.cpp -o Dynamic_Analysis

mkdir tmp
for dim in "x" "y" "z"; do
    python3 scripts/permuteDims.py host_code cu_code kernel dim "tmp/${dim}_${host_code##*.}" "tmp/${dim}_${cu_code##*.}"
    nvcc -c -arch=sm_20 -o "${dim}.o" $1
    g++ -o tracegen test.o trace.o -locelot
    ./tracegen > tmp/ocelot.trace # Not sure exactly what this command should be yet
    python3 scripts/clean_ocelot.py tmp/ocelot.trace "tmp/${dim}${kernel}.trace"
    if [[ $(./Dynamic_Analysis ${dim}) == *"Coalesced"* ]]; then
        mv "tmp/${dim}_${host_code##*.}" host_code
        mv "tmp/${dim}_${cu_code##*.}" cu_code
    fi

rm -r tmp