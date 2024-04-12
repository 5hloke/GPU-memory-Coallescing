# this is going to be our main function that will instrument the files and collect the traces. 

"""
Pseudocode: 
1. Instrument the cu file. 
2. Compile the cu file to ptx (reuqires nvcc)
3. Run the ptx file with the given input, pipe the output to a data structure.
4. Iterate over the data structure to collect the "dynamic id" parts of the output.
5. Write the (tx, ty, tz, static_id, dynamic_id, load/store, address) to a file.
"""

import subprocess
import sys
import os

class CudaRunner:
    def __init__(self, input_file, output_file, result_file):
        self.input_file = input_file
        self.output_file = output_file
        self.instrumented_file = 'instrumented.cu'
        self.compiled_file = 'compiled_instrumented.ptx'
        self.result_file = result_file

    def instrument_and_compile(self, compile_with_nvcc):
        # 1. Instrument the cu file using the instrument_cu.py script
        subprocess.run(["python", "instrument_cu.py", self.input_file, self.instrumented_file], check=True)
        
        # Optionally compile the cu file if compile_with_nvcc is True
        if compile_with_nvcc:
            # 2. Compile the cu file to ptx
            subprocess.run(["nvcc", self.instrumented_file, "-ptx", "-o", self.compiled_file], check=True)

    def run_and_extract(self):
        # 3. Run the ptx file with the given input, pipe the output to a data structure
        # TODO: The command and verifying it works
        process = subprocess.Popen(
            ["IDK HOW TO RUN A PTX FILE AND I AM AWAY FROM MY GPU :(", self.compiled_file],  
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = process.communicate()

        # Check for errors
        if process.returncode != 0:
            print(out.decode("utf-8"))
            raise Exception(err.decode("utf-8"))

        # 4. Iterate over the data structure to collect the "dynamic id" parts of the output
        output_data = self.parse_output(out.decode("utf-8"))

        # 5. Write the (tx, ty, tz, static_id, dynamic_id, load/store, address) to a file
        self.write_results(output_data)

    def parse_output(self, output):
        parsed_data = []
        dynamic_id_map = {}  # Map to keep track of the dynamic id for each (tx, ty, tz, static_id) tuple

        # Split the output into lines
        lines = output.splitlines()

        # Process each line to extract information
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) == 6 and parts[0] in ("ld", "st"):
                operation, tx, ty, tz, address, static_id = parts

                # Create a key for the dynamic_id_map with the relevant tuple information
                key = (tx, ty, tz, static_id) # these are going to be strings which is okay and makes it hashable

                # Check if key exists in the map, if not set to 0
                if key not in dynamic_id_map:
                    dynamic_id_map[key] = 0
                else:
                    # Increment the dynamic id for the static instruction
                    dynamic_id_map[key] += 1

                # Retrieve the dynamic id for the current line
                dynamic_id = dynamic_id_map[key]

                # Append the trace to the parsed data
                parsed_data.append((tx, ty, tz, static_id, dynamic_id, operation, address))

        return parsed_data

    def write_results(self, data):
        with open(self.result_file, "w") as f:
            for row in data:
                f.write(','.join(map(str, row)) + "\n")

if __name__ == "__main__":
    # Get input arguments for the runner (input file, output file, result file, whether to compile with nvcc)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    result_file = sys.argv[3]
    compile_with_nvcc = True if sys.argv[4].lower() == 'true' else False

    # Instantiate the runner and execute the process
    runner = CudaRunner(input_file, output_file, result_file)
    runner.instrument_and_compile(compile_with_nvcc)
    runner.run_and_extract()