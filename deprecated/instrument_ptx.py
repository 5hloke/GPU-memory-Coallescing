# instrument_ptx.py

import re
import sys
"""
Basically, we have a few things to do to instrument the PTX code:
1. Generate the strings that will be printed. static_id is different for each instruction.
2. Increase the number of registers. We use 4 b32 registers and 4 b64 registers for the vprintf calls. 
3. Insert the vprintf calls.

Sequencing it is a little tricky. Here's what we'l do: 
1. One pass to get the number of candidate loads and stores. Their sum is the number of registers we need to add. 
2. In the first pass we also save the line numbers of each load and store, as well as the line number of the last .global variable.
3. Then, once we have this information, we update the number of registers. 
"""

"""
Current TODO: 
1. Seems like the initial strings are being added in backwards order. This probably has to do with the insert_lines_helper sorting in reverse
So that there's no need to adjust the line numbers.
2. Registers are not actually getting updated. 
"""

def replace_lines_helper(list_of_strings_to_change, to_replace_list):
    # takes in a list of list of (line_num, line). 
    for list_of_lines in to_replace_list:
        for (line_num, new_line) in list_of_lines:
            list_of_strings_to_change[line_num] = new_line
    return list_of_strings_to_change

def insert_lines_helper(list_of_strings_to_change, to_insert_list):
    # takes in a list of (line_num, line[]) tuples. 
    # first sorts by line_num and then inserts the lines into ptx, keeping track of how many lines we've changed. 
    # Sort 'to_insert_list' by line numbers in descending order to avoid messing up indices
    to_insert_list.sort(key=lambda x: x[0], reverse=True)
    
    for line_num, lines_to_insert in to_insert_list:
        # Index is already 0-indexed; we can use 'line_num' directly to insert before it
        index_to_insert = line_num
        # Insert 'lines_to_insert' before 'index_to_insert'
        list_of_strings_to_change[index_to_insert:index_to_insert] = lines_to_insert

    return list_of_strings_to_change

def convert_to_ord(num : int):
    str_num = str(num)
    out = ""
    for char in str_num:
        out += str(ord(char)) + ", "
    # remove trailing space, comma of out
    return out[:-2]

class PTX_Function:    
    def __init__(self): # num registers before we dont care about actually. 
        self.number_r_registers_appearing_before = 0
        self.number_rd_registers_appearing_before = 0
        self.location_of_r_registers = 0
        self.location_of_rd_registers = 0
        self.load_lines_in_order = [] # (line_num, line, source_reg) tuple
        self.store_lines_in_order = []
        
class Instrumentor:
    def __init__(self, ptx_lines):
        self.ptx_lines = ptx_lines
        self.ptx_function_list = []
        self.original_number_of_strings = 0 # TODO: This might fuck up with local strings 
        self.last_global_line = None
        self.running_added_string_number = 0
        self.static_id = 0

    def parse_ptx(self):
        is_in_ptx_function = False
        register_pattern = re.compile(r"\s*.reg\s*.(b32|b64)\s*%(r|rd)<(\d+)>;")
        string_pattern = re.compile(r"$str($\d+)?\[\d+\]")

        for (line_num, line) in enumerate(self.ptx_lines):
            if ".global " in line or ".address_size " in line: # address_size will always be in ptx, global might not. TODO: Better way is to add a dummy global var to the source .cu code 
                self.last_global_line = line_num

            if string_pattern.match(line):
                self.original_number_of_strings += 1

            # actual instrumentation section below
            register_match = register_pattern.match(line)
            if "}" in line and is_in_ptx_function:
                self.ptx_function_list.append(ptx_function)
                is_in_ptx_function = False

            elif register_match:
                # create new ptx_function if not already in it (we'll have 2 matches for each ptx_function)

                if not is_in_ptx_function:
                    ptx_function = PTX_Function()
                    is_in_ptx_function = True
                
                if (register_match.group(2) == "r"):
                    ptx_function.number_r_registers_appearing_before = int(register_match.group(3)) # we know ptx_function has already been declared, dynamic typing to the rescue!
                    ptx_function.location_of_r_registers = line_num
                else: # must be rd register
                    ptx_function.number_rd_registers_appearing_before = int(register_match.group(3))
                    ptx_function.location_of_rd_registers = line_num
            elif "ld." in line:
                source_reg = line.split("[")[1].split("]")[0] # pull out [source_reg]

                if len(source_reg) < 10 and "SP" not in source_reg:
                    ptx_function.load_lines_in_order.append((line_num,line, source_reg))
            
            elif "st." in line:
                dest_reg = line.split("[")[1].split("]")[0] # pull out [dest_reg]
                if len(dest_reg) < 10 and "SP" not in dest_reg:
                    ptx_function.store_lines_in_order.append((line_num,line, dest_reg))
        
        if self.last_global_line is None:
            raise Exception("Placehodler: No .global variable found in PTX code. Please add a dummy global variable to the source .cu file.")

    def insert_global_strings(self):
        pass

    def update_registers_for_this_ptx_function(self, func):
        to_replace = [] # line_num, line
        r_register_declaration_line = "\t.reg .b32 	%r<69>;" # TODO: brittle af
        rd_register_declaration_line = "\t.reg .b32 %rd<420>;" # TODO: brittle af, also do the spaces matter for nvidia? guess we'll find out in testing cause their fucking compiler is fucking closed source!

        after_number_r_registers = func.number_r_registers_appearing_before + 4 * (len(func.load_lines_in_order) + len(func.store_lines_in_order))
        after_number_rd_registers = func.number_rd_registers_appearing_before + 4 * (len(func.load_lines_in_order) + len(func.store_lines_in_order))

        changed_r_register_line = re.sub(r"<\d+>", f"<{str(after_number_r_registers)}>", r_register_declaration_line)
        to_replace.append((func.location_of_r_registers, changed_r_register_line))

        changed_rd_register_line = re.sub(r"<\d+>", f"<{str(after_number_rd_registers)}>", rd_register_declaration_line)
        to_replace.append((func.location_of_rd_registers, changed_rd_register_line))

        return to_replace
    
    def get_vprintf_lines(self, actual_inst, address_register, r_running_number_of_registers, rd_running_number_of_registers, string_identifier):
        output_lines = []
        # Load threadIdx and the address into appropriate registers
        output_lines.append(f"// Set up registers r{r_running_number_of_registers}-r{r_running_number_of_registers + 3} with threadIdx.x, threadIdx.y, threadIdx.z\n")
        output_lines.append(f"mov.u32 %r{r_running_number_of_registers}, %tid.x;\n")
        output_lines.append(f"mov.u32 %r{r_running_number_of_registers+1}, %tid.y;\n")
        output_lines.append(f"mov.u32 %r{r_running_number_of_registers+2}, %tid.z;\n")
        
        # rd51 will be used to store the address, ensure it's 64 bits
        output_lines.append(f"// Convert the 32-bit address in {address_register} to 64-bit and place it in rd{rd_running_number_of_registers}\n")
        output_lines.append(f"cvt.u64.u32 %rd{rd_running_number_of_registers}, {address_register};\n")

        # return output_lines.append(        )
        # dump the string params to the stack
        output_lines.append(f"st.u32 	[%SP+0], %r{r_running_number_of_registers};\n") # store tid.x)
        output_lines.append(f"st.u32 	[%SP+4], %r{r_running_number_of_registers+1};\n") # store tid.y)
        output_lines.append(f"st.u32 	[%SP+8], %r{r_running_number_of_registers+2};\n") # tid.z)
        output_lines.append(f"st.u64 	[%SP+16], %rd{rd_running_number_of_registers};\n") # address)

        # This ptx_function assumes that the string literal address and the register values
        # have already been placed correctly in the registers and on the stack
        output_lines.append(f"mov.u64 %rd{rd_running_number_of_registers+1}, $str{{{string_identifier}}}$;\n")  # Placeholder for actual string literal name TODO I guess since we're instrumenting the cuda to insert a dummy string this will always be more than 1)
        output_lines.append(f"add.u64 %rd{rd_running_number_of_registers+2}, %SP, 0;\n")  # Assuming SP is pointing to the right place)
        output_lines.append(f"add.u64 %rd51, %SP, 0;\n")  # Adjust address as necessary TODO this one right here might be copilot hallucinating)

        # this is the actual print call
        output_lines.append("{\n")
        output_lines.append("    .reg .b32 temp_param_reg;\n")
        output_lines.append("    .param .b64 param0;\n")
        output_lines.append("    .param .b64 param1;\n")
        output_lines.append("    .param .b32 retval0;\n")
        output_lines.append(f"    st.param.b64 [param0+0], %rd{rd_running_number_of_registers+1};\n")  # String literal)
        output_lines.append(f"    st.param.b64 [param1+0], %rd{rd_running_number_of_registers+2};\n") # Stack pointer with arguments)
        output_lines.append("    call.uni (retval0), \n")
        output_lines.append("    vprintf, \n")
        output_lines.append("    (\n")
        output_lines.append("    param0, \n")
        output_lines.append("    param1\n")
        output_lines.append("    );\n")
        output_lines.append("    ld.param.b32 %r52, [retval0+0];\n")  # Optional? Store vprintf return value)
        output_lines.append("}\n")

        return output_lines


    def get_string_number_and_string(self, register: str, is_load: bool):
        # create the .global string literal. Return the string number that it is.  
        this_string_number = self.running_added_string_number + self.original_number_of_strings
        string_to_return = None
        if is_load:
            string_to_return = f".global .align 1 .b8 $str${this_string_number}[{20 + len(str(self.static_id))}] = {{108, 100, 44, 32, 37, 100, 44, 32, 37, 100, 44, 32, 37, 100, 44, 32, 37, 112, {convert_to_ord(self.static_id)}, 0}};\n"
        else: 
            string_to_return = f".global .align 1 .b8 $str${this_string_number}[{20 + len(str(self.static_id))}] = {{115, 116, 44, 32, 37, 100, 44, 32, 37, 100, 44, 32, 37, 100, 44, 32, 37, 112, {convert_to_ord(self.static_id)}, 0}};\n"

        self.running_added_string_number += 1
        return this_string_number, string_to_return



    def insert_vprintf_calls_and_modify_number_of_registers(self):
        global_to_replace = [] # list of (line_num, line)
        global_to_insert = [] # list of (line_num, lines[])

        for ptx_function in self.ptx_function_list:
            # update registers
            global_to_replace.append(self.update_registers_for_this_ptx_function(ptx_function)) # takes in the location of r, rd registers. Increases them each by 4 * number of loads and stores.
            # insert vprintf calls
            r_running_number_of_registers = ptx_function.number_r_registers_appearing_before
            rd_running_number_of_registers = ptx_function.number_rd_registers_appearing_before

            for (line_num, line, source_reg) in ptx_function.load_lines_in_order:
                
                running_string_number, string_to_insert = self.get_string_number_and_string(source_reg, True)
                global_to_insert.append((running_string_number, [string_to_insert])) # +1 since we insert just after the last global line, our insert helper will insert before the required line.
                global_to_insert.append((line_num, self.get_vprintf_lines(line, source_reg, r_running_number_of_registers, rd_running_number_of_registers, running_string_number)))
                self.static_id += 1
                
            for (line_num, line, dest_reg) in ptx_function.store_lines_in_order:
                running_string_number, string_to_insert = self.get_string_number_and_string(dest_reg, False)
                global_to_insert.append((running_string_number, [string_to_insert])) # +1 since we insert just after the last global line, our insert helper will insert before the required line.
                global_to_insert.append((line_num, self.get_vprintf_lines(line, dest_reg, r_running_number_of_registers, rd_running_number_of_registers, running_string_number)))
                self.static_id += 1

        # replace necessary lines
        print(len(self.ptx_lines))
        self.ptx_lines = replace_lines_helper(self.ptx_lines, global_to_replace)
        print(len(self.ptx_lines))
        

        # insert necessary lines
        print(len(self.ptx_lines))

        self.ptx_lines = insert_lines_helper(self.ptx_lines, global_to_insert)
        print(len(self.ptx_lines))

    def write_ptx_to_file(self, filename):
        with open(filename, 'w') as f:
            f.writelines(self.ptx_lines)

def main(input_file, output_file): 
    with open(input_file, "r") as f: 
        ptx_lines = f.readlines()

    instrumentor = Instrumentor(ptx_lines)

    instrumentor.parse_ptx()

    # globals_lines_line_number_pairs = [] # requires: know all instructions whether load or store, and a static identifier. know the last lijne of the .global variables.

    # instrumentor.insert_strings(globals_lines_line_number_pairs)

    instrumentor.insert_vprintf_calls_and_modify_number_of_registers()

    instrumentor.write_ptx_to_file(output_file)



if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        # If not, print the proper usage of the script and exit
        print("Usage: python script.py <input_file> <output_file>")
        sys.exit(1)

    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]

    main(input_file_name, output_file_name) # example python instrument_ptx.py basic_simpler.ptx instrumented.ptx