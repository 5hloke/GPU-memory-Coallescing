#!/usr/bin/env python3
import sys

def insert_trace_call(file_path, main_function):
    with open(file_path, 'r+') as file:
        content = file.readlines()
        main_found = False
        for i, line in enumerate(content):
            if f"int main()" in line:
                main_found = True
            if main_found and '{' in line:
                # Insert the external function declaration and the call after the main function starts
                content.insert(i+1, f"    {main_function}(); // add this\n")
                content.insert(0, f"extern void {main_function}(); // add this\n")
                break
        file.seek(0)
        file.writelines(content)

if __name__ == "__main__":
    insert_trace_call(sys.argv[1], sys.argv[2])