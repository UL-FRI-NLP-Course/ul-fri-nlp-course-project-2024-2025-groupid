import json
import re

def txt_to_json(input_file, output_file):
    # Read the input text file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split the content into prompt and answers sections
    parts = re.split(r'ODGOVORI:', content, flags=re.IGNORECASE)
    if len(parts) < 2:
        raise ValueError("The text file must contain 'ODGOVORI:' to separate prompt from answers")
    
    prompt = parts[0].strip()
    answers_section = parts[1].strip()
    
    # Parse the answers
    answers = {}
    answer_lines = answers_section.split('\n')
    
    for line in answer_lines:
        if not line.strip():
            continue
        
        # Split model name from answer
        model_match = re.match(r'^(.*?):\s*(.*)', line)
        if model_match:
            model = model_match.group(1).strip().lower()
            answer = model_match.group(2).strip()
            answers[model] = answer
    
    # Create the JSON structure
    json_data = {
        "prompt": prompt,
        "odgovori": {
            "GaMs-1B": answers.get("GaMs-1B".lower(), ""),
            "GaMs-9B": answers.get("GaMs-9B".lower(), ""),
            "Llama 3.1 8B": answers.get("Llama 3.1 8B".lower(), ""),
            "mt0-large": answers.get("mt0-large".lower(), "")
        }
    }
    
    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    
    print(f"Successfully converted {input_file} to {output_file}")

# Example usage:
input_filename = 'input.txt'
output_filename = 'output.json'

txt_to_json(input_filename, output_filename)