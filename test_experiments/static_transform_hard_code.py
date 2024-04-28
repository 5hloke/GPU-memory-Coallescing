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
N = sys.argv[7]



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
    pattern = r"__global__\s*void\s*" + re.escape(kernel)

    # Use re.search() to find the pattern
    match = re.search(pattern, code)

    if match:
        return match.start()
    else:
        return -1
    
    
def get_op_and_global_addrs(fn):
    # This should match anything of the reduction form y[i] += A[j] * x[k] given in the paper and get y, plus/minus, A, k
    pattern = r'\s*([a-zA-Z0-9_]+)\[[0-9]+\]\s*([\+-])=\s*([a-zA-Z0-9_]+)\[[a-zA-Z0-9_]+\]\s*\*\s*([a-zA-Z0-9_]+)\[[a-zA-Z0-9_]+\]\s*;'
    matches = re.findall(pattern, fn)

    # Assume just one match
    return matches[0]


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
    
    y, pm, A, x = tuple(get_op_and_global_addrs(fn))

    fn0 = fn[0] + "__shared__ float 583_shared[blockDim.x * blockDim.x];\n"
    #S[i*blockDim.x+j] = A[index1]
    fn1 ="583_i =threadIdx.x/blockDim.x;\n583_j = threadIdx.x %% blockDim.x;\n"
    fn2 = "583_block_id = blockIdx.x * blockDim.x + blockIdx.y;\n"
    fn3 = "583_t1 = 583_block_id * blockDim.x * blockDim.x;\n 583_t2 = (583_block_id * blockDim.x) %% "+str(N)+";\n"
    fn4 = "583_index1= 583_t1 + 583_t2 + 583_i * "+str(N)+" + 583_j;\n"
    fn5 = "583_index2 = (583_block_id * blockDim.x + 583_j) %% " + str(N) + ";\n"
    fn6 = "583_index3 = (583_block_id * blockDim.x * blockDim.x) / " + str(N) + " + 583_i;\n" 
    fn7 ="583_shared[583_i*blockDim.x + 583_j] = " + A + "[583_index1];\n"
    fn8 = "583_shared[583_i*blockDim.x + 583_j] *= " +  x + "[583_index2];\n__syncthreads();\n"
    fn = fn[1:]
    pattern = r'(?:[;\s])for(?:\s|()'
    matches = re.findall(pattern, fn)[0][0]
    fn9 = fn[:matches]
    fn = fn[matches:]
    fn_begin = fn.find("{") +1
    fn_end = find_closing_bracket(fn[fn_begin:])+ fn_begin

    # Need to create the entire string for the new for loop
    fn10 = "for(int k=0;k<blockDim.x;k++)\{"+ y +" [583_i] += 583_shared[583_i * blockDim.x + k];\}\n"

    fn = fn[fn_end:]





    
    # fn = fn.replace("blockDim.x", "$$$$$$$$$PLACEHOLDER$$$$$$$$$$")
    # fn = fn.replace("blockDim.{0}".format(lead_dim), "blockDim.x")
    # fn = fn.replace("$$$$$$$$$PLACEHOLDER$$$$$$$$$$", "blockDim.{0}".format(lead_dim))
    return code[:fn_begin] + fn0 + fn1+fn2+fn3+fn4+fn5+fn6+fn7+fn8+fn9+fn10+fn + code[fn_end:]


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
        