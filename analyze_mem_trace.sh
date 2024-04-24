#!/bin/bash

# Check for the minimum number of arguments
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 main_function [file] [--custom-trace custom_trace_file]"
    exit 1
fi

main_function="$1"
file="${2:-}"

# Function to compile and run the trace generation
compile_and_run() {
    echo "Compiling with g++ for trace.cpp..."
    g++ -std=c++0x src/trace.cpp -c -l
    echo "Compiling with nvcc for $file..."
    nvcc -c -arch=sm_20 "${file}.cu"
    echo "Linking and generating executable..."
    g++ -o tracegen "${file}.o" custom_trace.o -locelot
    echo "Running trace generation..."
    ./tracegen > ocelot_output.txt
    echo "Cleaning ocelot output..."
    python3 scripts/clean_ocelot.py ocelot_output.txt cleaned_ocelot_output.txt
    # echo "Running dynamic analysis..."
    # g++ -std=c++0x dynamic_analysis.cpp -o dynamic_analysis
    # ./dynamic_analysis.o cleaned_ocelot_output.txt
}

# Check for the --custom-trace option and set the custom_trace_file variable
custom_trace_flag="$3"
custom_trace_file="$4"

# Modify custom_trace.cpp and compile
# if [[ "$custom_trace_flag" == "--custom-trace" ]] && [[ -n "$custom_trace_file" ]]; then
# echo "Custom trace file specified: $custom_trace_file"
#     # Call the Python script to insert the trace call
#     ./insert_trace_call.py "$custom_trace_file" "$main_function"
# else
#     # No custom trace file specified, modify the default trace file
#     echo "No custom trace file specified. Modifying default trace file."
#     python3 insert_trace_call.py "custom_trace.cpp" "$main_function"
# fi

# Compile and run if a file is specified
if [[ -n "$file" ]]; then
    compile_and_run
else
    echo "No file specified for compilation. Exiting."
    exit 1
fi
