import re
import sys

"""Script for permuting the dimensions of CUDA code via string search/replace.
Command line args: <host code file> <.cu file> <leading dimension \in {"x", "y", "z"}
"""

if sys.argv[4] == 'x':
    exit

# Permute dims in host code launch:
with open(sys.argv[1]) as code:
    with open("lead" + sys.argv[4] + "_" + sys.argv[1], "w") as new_file:
        for line in code.readlines():
            is_block = line.startswith("dim3 dimBlock(")
            is_grid = line.startswith("dim3 dimGrid(")
            if is_block or is_grid:
                # line = line[line.find('(') + 1:line.find(')')]
                # line_split = line.split(',')
                
                # if sys.argv[4] == 'y':
                #     line_split[0], line_split[1] = line_split[1], line_split[0]
                # elif sys.argv[4] == 'z':
                #     line_split[0], line_split[2] = line_split[2], line_split[0]
                    
                # line = ", ".join(line_split) + ");"
                
                # if is_block:
                #     line = "dim3 dimBlock(" + line
                # else:
                #     line = "dim3 dimGrid(" + line
                
                pattern = r'\((\w+), (\w+), (\w+)\)'
                if sys.argv[4] == 'y':
                    replacement = r'(\2, \1, \3)'
                elif sys.argv[4] == 'z':
                    replacement = r'(\3, \2, \1)'
                else:
                    raise ValueError("Dim must be x, y, or z")
                line = re.sub(pattern, replacement, line)
                
            new_file.write(line + "\n")
            
with open(sys.argv[2]) as code:
    with open("lead" + sys.argv[4] + "_" + sys.argv[2], "w") as new_file:
        code = code.read()
        pattern = r'(%tid)\.(x|' + sys.argv[4] + ')'

        # Define the replacement pattern with captured groups swapped
        replacement = lambda match: match.group(1) + '.'  + (sys.argv['4'] if match.group(2) == 'x' else 'x')

        # Perform the find and replace using regex
        new_text = re.sub(pattern, replacement, code)
        new_file.write(new_text)