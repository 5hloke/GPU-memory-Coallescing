# #!/usr/bin/env python3
# import sys

# def insert_trace_call(file_path, main_host_function):
#     with open(file_path, 'r+') as file:
#         content = file.readlines()
#         main_found = False
#         idx_of_main_function = 0
#         idx_of_end_of_main_function = 0
#         for i, line in enumerate(content):
#             if f"int main()" in line:
#                 main_found = True
#                 idx_of_main_function = i
#             if main_found and '}' in line:
#                 # Insert the external function declaration and the call after the main function starts
#                 idx_of_end_of_main_function = i
#                 break
#         file.seek(0)

#         content.insert(idx_of_end_of_main_function, f"\t{main_host_function}(); // added via script\n")
#         content.insert(idx_of_main_function, f"extern void {main_host_function}(); // added via script\n")
#         file.writelines(content)

# if __name__ == "__main__":
#     # insert_trace_call(sys.argv[1], sys.argv[2])


#!/usr/bin/env python3
import sys

def insert_trace_call(file_path, main_host_function):
    with open(file_path, 'r+') as file:
        content = file.readlines()
        main_found = False
        idx_of_main_function = 0
        idx_of_end_of_main_function = 0
        # First pass: Find main function and remove old script additions
        new_content = []
        for line in content:
    
            if "// added via script" not in line:
                new_content.append(line)
                
            if "int main()" in line:
                main_found = True
                idx_of_main_function = len(new_content) - 1
        # Second pass: Find the end of the main function
        for i, line in enumerate(new_content[idx_of_main_function:], start=idx_of_main_function):
            if main_found and '}' in line:
                idx_of_end_of_main_function = i
                break
        # Insert the external function declaration and the call
        # new_content.insert(idx_of_end_of_main_function, f"\t{main_host_function}(); // added via script\n")
        # new_content.insert(idx_of_main_function, f"extern void {main_host_function}(); // added via script\n")
        new_content.insert(idx_of_end_of_main_function, "\t{}(); // added via script\n".format(main_host_function))
        new_content.insert(idx_of_main_function, "extern void {}(); // added via script\n".format(main_host_function))
        file.seek(0)
        file.truncate()
        file.writelines(new_content)

if __name__ == "__main__":
    insert_trace_call(sys.argv[1], sys.argv[2])
