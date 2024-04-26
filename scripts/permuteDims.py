import sys
import re

"""Script for permuting the dimensions of CUDA code via string search/replace.
Command line args: <host code file> <.cu file> <name of kernel> <leading dimension \in {"x", "y", "z"}> <new host output> <new .cu output>
"""

host_code = sys.argv[1]
cu_code = sys.argv[2]
kernel = sys.argv[3]
lead_dim = sys.argv[4]
new_host = sys.argv[5]
new_cu = sys.argv[6]


def transform_host(code):
    output = ""
    for line in code:
        if line.lstrip().startswith(kernel + "<<<"):
            dims = line.lstrip()[len(kernel) + 3:]
            dims = dims[:dims.find(">>>")].split(',')
            grid_name, block_name = dims[0].strip(), dims[1].strip()
            # output += "std::swap({0}.x, {0}.{1});\n".format(grid_name, lead_dim)
            output += "std::swap({0}.x, {0}.{1});\n".format(block_name, lead_dim)
            output += line
            # output += "std::swap({0}.x, {0}.{1});\n".format(grid_name, lead_dim)
            output += "std::swap({0}.x, {0}.{1});\n".format(block_name, lead_dim)
        else:
            output += line
                
    return output


def find_closing_bracket(s):
    # Takes a str s which begins with a open { and finds its close
    assert s[0] == "{"
    bracket_count = 0
    
    for i, c in enumerate(s):
        if c == "{":
            bracket_count += 1
        elif c == "}":
            bracket_count -= 1
            if bracket_count == 0:
                return i
            
def find_kernel_usage(code):
    pattern = r"void\s*" + re.escape(kernel)

    # Use re.search() to find the pattern
    match = re.search(pattern, code)

    if match:
        return match.start()
    else:
        return -1


def transform_cu(code):
    kernel_loc = find_kernel_usage(code)
    if not kernel_loc:
        raise ModuleNotFoundError("Could not find a reference to the kernel in given .cu file")
    
    while kernel_loc != -1:
        statement = code[kernel_loc:code[kernel_loc:].find(";") + kernel_loc]
        fn_begin = statement.find("{")
        if fn_begin != -1:
            fn_begin += kernel_loc
            fn_end = find_closing_bracket(code[fn_begin:]) + fn_begin
            break
        kernel_loc = find_kernel_usage(code[kernel_loc + 1:]) + kernel_loc + 1
        
    fn = code[fn_begin:fn_end]
    # fn = fn.replace("threadIdx.x", "$$$$$$$$$PLACEHOLDER$$$$$$$$$$")
    # fn = fn.replace("threadIdx.{0}".format(lead_dim), "threadIdx.x")
    # fn = fn.replace("$$$$$$$$$PLACEHOLDER$$$$$$$$$$", "threadIdx.{0}".format(lead_dim))
    return code[:fn_begin] + fn + code[fn_end:]


if host_code == cu_code:
    with open(host_code) as code:
        code_lines = code.readlines()
        with open(new_host, "w") as f:
            f.write(transform_cu(transform_host(code_lines)))
else:
    with open(host_code) as code:
        code_lines = code.readlines()
        with open(new_host, "w") as f:
            f.write(transform_host(code_lines))

    with open(cu_code) as code:
        with open(new_cu, "w") as f:
            f.write(transform_cu(code.read()))
        