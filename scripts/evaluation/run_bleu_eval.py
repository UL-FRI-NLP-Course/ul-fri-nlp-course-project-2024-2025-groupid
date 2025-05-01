import subprocess
import re
import os
from tempfile import mkstemp
from shutil import move

def modify_smoothing_method(file_path, method_num):
    # Create temp file
    fh, abs_path = mkstemp()
    
    with os.fdopen(fh, 'w', encoding='utf-8') as new_file:
        with open(file_path, 'r', encoding='utf-8') as old_file:
            for line in old_file:
                if 'smooth_fn = SmoothingFunction().method' in line:
                    new_line = f'        smooth_fn = SmoothingFunction().method{method_num}\n'
                    new_file.write(new_line)
                else:
                    new_file.write(line)
    
    os.remove(file_path)
    move(abs_path, file_path)

def run_bleu_evaluations():
    script_path = 'calc_bleu.py'
    original_content = open(script_path, 'r', encoding='utf-8').read()
    
    # Loop through smoothing methods
    for method in range(1, 8):
        method_results = {}
        
        # Modify the script with current method
        modify_smoothing_method(script_path, method)
        
        # Loop through prompts
        for prompt in range(1, 8):
            # Run calc_bleu.py script
            cmd = f"py {script_path} --file prompt{prompt}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            # Extract the relevant part of the output
            output = result.stdout
            relevant_output = extract_relevant_output(output)
            
            # Store the results
            method_results[f"prompt{prompt}"] = relevant_output
        
        # Save results
        save_method_results(method, method_results)
    
    # Restore original file content
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(original_content)

def extract_relevant_output(full_output):
    lines = full_output.split('\n')
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        if "Model Performance Summary:" in line:
            start_idx = i + 1
        if start_idx is not None:
            end_idx = len(lines)
            break
    
    if start_idx is not None and end_idx is not None:
        return '\n'.join(lines[start_idx:end_idx+1])
    return ""

def save_method_results(method_num, results):
    filename = f"results_method{method_num}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        for prompt, result in results.items():
            f.write(f"{prompt}:\n{result}\n")
    print(f"Saved results for method {method_num} to {filename}")

if __name__ == "__main__":
    run_bleu_evaluations()