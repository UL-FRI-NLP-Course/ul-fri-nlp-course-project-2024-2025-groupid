import json
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from difflib import SequenceMatcher

def load_data(json_file):
    """
    Load data from JSON file
    
    json_file: File containing the data

    return: Data from the file
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def find_matching_ground_truth(example_text, io_data):
    """
    Find the most similar ground truth output from io.json for a given example.

    example_text: The full example text
    io_data: List of input-output pairs from io.json

    return: The matching ground truth output or None if not found
    """
    best_match = None
    best_ratio = 0

    for item in io_data:
        io_input = item['input'].strip()
        ratio = SequenceMatcher(None, example_text, io_input).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = item['output']
    
    # print("example TEXT: ", example_text, '\n')
    # print("BEST MATCH: ", best_match)

    return best_match

def calculate_bleu(model_outputs, references):
    """
    Calculate BLEU scores for model outputs against references
    
    model_outputs: List of model-generated texts
    references: List of reference texts
    
    return: Dictionary with various BLEU scores
    """
    # Calculate sentence-level BLEU scores
    sentence_scores = []
    smooth_fn = SmoothingFunction().method1
    for candidate, refs in zip(model_outputs, references):
        score = sentence_bleu(refs, candidate, smoothing_function=smooth_fn)
        sentence_scores.append(score)
    
    # Calculate corpus-level BLEU scores
    corpus_score = corpus_bleu(references, model_outputs, smoothing_function=smooth_fn)
    
    return {
        'sentence_scores': sentence_scores,
        'average_sentence_score': sum(sentence_scores) / len(sentence_scores),
        'corpus_score': corpus_score
    }

def evaluate_examples(examples_file, io_file):
    """
    Evaluate model outputs against ground truth
    """
    # Load examples and io data
    examples_data = load_data(examples_file)

    print(f"Loaded data from {examples_file.split("/")[-1]} file")

    io_data = load_data(io_file)
    
    # Prepare results structure
    results = {
        'models': {},
        'examples': []
    }
    
    # Initialize model structures
    for model_name in next(iter(examples_data))['odgovori'].keys():
        results['models'][model_name] = {
            'all_scores': [],
            'all_references': [],
            'all_candidates': []
        }
    
    # Process each example
    for example_idx, example_item in enumerate(examples_data, 1):
        example_text = example_item['prompt']
        ground_truth = find_matching_ground_truth(example_text, io_data)
        
        if not ground_truth:
            print(f"Warning: No matching ground truth found for example {example_idx}")
            continue
        
        example_result = {
            'example_id': example_idx,
            'ground_truth': ground_truth,
            'model_scores': {}
        }
        
        # Evaluate each model's response
        for model_name, model_output in example_item['odgovori'].items():
            # Tokenize
            reference = [word_tokenize(ground_truth.lower())]
            candidate = word_tokenize(model_output.lower())
            
            # Calculate BLEU
            score = sentence_bleu(reference, candidate)
            
            # Store results
            example_result['model_scores'][model_name] = score
            results['models'][model_name]['all_scores'].append(score)
            results['models'][model_name]['all_references'].append(reference)
            results['models'][model_name]['all_candidates'].append(candidate)
        
        results['examples'].append(example_result)
    
    # Calculate corpus-level scores for each model
    for model_name, model_data in results['models'].items():
        if model_data['all_references'] and model_data['all_candidates']:
            model_data['corpus_bleu'] = corpus_bleu(
                model_data['all_references'],
                model_data['all_candidates']
            )
            model_data['average_bleu'] = sum(model_data['all_scores']) / len(model_data['all_scores'])
    
    return results

def print_results(results):
    """
    Print evaluation results in a readable format
    """
    print("BLEU Score Evaluation Results")
    print("============================")
    
    # Print corpus-level results for each model
    print("\nModel Performance Summary:")
    print("{:<15} {:<15} {:<15}".format("Model", "Avg BLEU", "Corpus BLEU"))
    print("-" * 45)
    for model_name, model_data in results['models'].items():
        avg = model_data.get('average_bleu', 0)
        corpus = model_data.get('corpus_bleu', 0)
        print("{:<15} {:<15.4f} {:<15.4f}".format(model_name, avg, corpus))
    
    # Print detailed example-level results
    print("\nDetailed example Results:")
    for example in results['examples']:
        print(f"\nexample {example['example_id']}:")
        print("Ground Truth:", example['ground_truth'][:100] + "...")
        print("{:<15} {:<10}".format("Model", "BLEU"))
        print("-" * 25)
        for model_name, score in example['model_scores'].items():
            print("{:<15} {:<10.4f}".format(model_name, score))

if __name__ == "__main__":
    """
    Use:

    python calc_bleu.py --file <file_name>

    Parameters:
        --file
        Select the file with the prompts to use
        Options:
            - prompt1
            - prompt2
            - prompt3
            - prompt4
            - prompt5
            - prompt6
            - prompt7
    """
    import argparse
    parser = argparse.ArgumentParser(description='Evaluation of prompts')
    parser.add_argument('--file', type=str, default='prompt1', help='File with prompts to use')
    args = parser.parse_args()

    # Check if prompt file is given
    if args.file:
        file_name = args.file
    else:
        file_name = 'prompt1'
    
    eval_path = '../../data/evaluation'
    results = evaluate_examples(f'{eval_path}/{file_name}.json', f'{eval_path}/io.json')
    print_results(results)