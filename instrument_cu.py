# instrument_cu.py

"""
Insert a global string and dummy printf function to the main method in the CU code. 
"""
import sys
def insert_print_statement(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    main_found = False
    brace_found = False
    for i, line in enumerate(lines):
        # Check for the main function signature
        if 'int main' in line and ('(' in line and ')' in line):
            main_found = True
        if main_found and '{' in line:
            # Found the opening brace after the main signature, set the flag
            brace_found = True
            main_found = False
            insert_point = i
            break
    
    if brace_found:
        # Insert the printf statement after the opening brace line
        lines.insert(insert_point + 1, '\tprintf("This code has been instrumented\\n");\n')
        
        # Write the modified code to the output file
        with open(output_file_path, 'w') as file:
            file.writelines(lines)
        print("Instrumentation complete – code has been modified and saved to", output_file_path)
    else:
        print("Could not find the main function or the opening brace in the file.")

# Usage example


if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        # If not, print the proper usage of the script and exit
        print("Usage: python script.py <input_file> <output_file>")
        sys.exit(1)

    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]

    insert_print_statement(input_file_name, output_file_name) # example python instrument_ptx.py basic_simpler.ptx instrumented.ptx
# insert_print_statement('input.cu', 'output.cu')