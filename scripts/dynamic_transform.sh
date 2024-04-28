#!/bin/sh
# set -euxo pipefail

# Script for running profiling and outputting the best geometry as a new kernel
# Currently only works if you name your .cu's main entry function as sampleKernel

if [ $# -eq 4 ]; then
    code=$1
    out=$2
    main=$3
    kernel=$4
else
    echo "Usage: bash dynamic_transform.sh <input kernel file> <output kernel file> <main name> <kernel name>"
    exit 1
fi

if [ code = out ]; then
    echo "input and output filenames must be different"
    exit 1
fi

g++ -std=c++11 src/trace.cpp -c # should already be compiled
g++ -std=c++11 src/Dynamic_Analysis.cpp -o Dynamic_Analysis # should already be compiled

mkdir tmp
for dim in "x" "y" "z"; do
    permutedfile="tmp/${dim}.${code##*.}"
    echo "Permuting geometry of kernel: ${kernel} in ${code}..."
    echo "Leading with dimension ${dim}..."
    python3 scripts/permuteDims.py ${code} ${code} ${kernel} ${dim} ${permutedfile} ${permutedfile}
    echo "Finished permuting, result stored in ${permutedfile}"

    outfile="tmp/${dim}_trace.txt"
    echo "Calling trace analysis script..."
    bash analyze_mem_trace.sh ${main} ${permutedfile} ${out_file}
    echo "Finished tracing, result stored in ${outfile}"

    echo "Running dynamic analysis on ${permutedfile}..."
    if [[ $(./Dynamic_Analysis $$) == *"Coalesced"* ]]; then
        echo "Memory was coalesced!"
        mv ${permutedfile} ${out}
    else
        echo "Not coalesced"
    fi
done

rm -r tmp
exit 0

# nvcc -c -arch=sm_20 -o "${dim}.o" $1
# g++ -o tracegen "${dim}.o" trace.o -locelot
# ./tracegen > tmp/ocelot.trace # Not sure exactly what this command should be yet
# python3 scripts/clean_ocelot.py tmp/ocelot.trace "tmp/${dim}${kernel}.trace"