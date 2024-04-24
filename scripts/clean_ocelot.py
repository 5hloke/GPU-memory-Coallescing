import sys
import re

def main(file_url, outfile_url):

    with open(file_url, "r") as f:
        lines = f.readlines()

        lines_modified = []
        kept_static_ids_set = set()
        current_kept_static_id = 0

        dynamic_id_map = {}
        trace_pattern = re.compile(r"(\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (ld|st)\.global\.(\S+) %\S+, \[%\S+\] (\d+)")

        for line in lines:
            match = trace_pattern.match(line)
            if match is not None:
                grid_x, grid_y, grid_z, tid_x, tid_y, tid_z, static_id, instruction, _dtype, address = match.groups()
                
                if static_id not in kept_static_ids_set:
                    kept_static_ids_set.add(static_id)
                    current_kept_static_id += 1
                
                dynamic_id_key = f"{current_kept_static_id}.{tid_x}.{tid_y}.{tid_z}"

                if dynamic_id_key not in dynamic_id_map:
                    dynamic_id_map[dynamic_id_key] = 0
                else:
                    dynamic_id_map[dynamic_id_key] += 1

                # print(current_kept_static_id, dynamic_id_key)
                is_ld = 0 if instruction == "ld" else 1

                result = f"{grid_x} {grid_y} {grid_z} {tid_x} {tid_y} {tid_z} {is_ld} {current_kept_static_id} {dynamic_id_map[dynamic_id_key]} {address}\n"

                lines_modified.append(result)

    with open(outfile_url, "w") as f:
        f.writelines(lines_modified)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py inputfile.txt outputfile.txt")
    else:
        main(sys.argv[1], sys.argv[2])
