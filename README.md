# 583-final-project

Authors: Shlok Agarwal, Karan Anand, James Edwards, Finn Roblin, and Arnav Shah

This is the repo for our University of Michigan EECS 583: Advanced Compilers final project. Our work is based off Fauzia et al.'s 2015 paper [Characterizing and enhancing global memory data coalescing on GPUs](https://ieeexplore.ieee.org/document/7054183). From this paper, we successfully replicated the tracing and dynamic analysis tools as well as the transformation they give which uses the output of the dynamic analysis to modify the thread geometry of a cuda program in order to improve global memory coalescence.

## Repo Structure

### Directories
- **/scripts** - Contains shell scripts to automatically run dynamic analysis, as well as python parsers to transform cuda files.
- **/src** - Contains cpp source files for project.
- **/test_experiments** - Contains cuda and ptx files used for experiments.

### Files
- **scripts/dynamic_transform.sh** - The main entry point to the tool. Runs profiling and outputs the most performant geometry as a new kernel. See file for more details
- **commands_to_run_ocelot.md** - Explanation of commands that may be necessary to grab ocelot docker image and set it up in the same manner as this paper.
- **scripts/clean_ocelot** - Transforms an ocelot trace output into format suitable for the dynamic analysis program.
- **scripts/insert_trace_call.py** - Modifies profiling program with the kernel name to instrument.
- **scripts/permuteDims.py** - Permutes kernel thread geometry.
- **src/Dynamic_Analysis.cpp** - Run dynamic analysis on trace file, report number coalesced/uncoalesced accesses.
- **src/static_transform.cpp** - A partial implementation of a GPUOcelot transformation pass, before we realized that the analysis tools are broken.
- **src/trace.cpp** - The GPUOcelot tracing function.
- **test_experiments/gaussian.cu** - Implementation of Gaussian elimination (turn matrix into row reduced form).
- **test_experiments/static_transform_reduction.py** - Attempt to implement shared memory static transformation.
- **test_experiments/test.cu** - Stencil computation taking the median across 4 neighboring cells.
- **test_experiments/test.ptx** - PTX code of test.cu.
- **test_experiments/test_reduction.cu** - Array-vector multiplication, which is technically a reduction.
- **583demo.mp4** - The video shown in our demo presentation. Please note that the commands are run individually for clarity only, and scripts/dynamic_transform.sh is the main entry point to using our work.
- **analyze_mem_trace.sh** - Utility used by **scripts/dynamic_transform.sh** to generate and analyze traces.
