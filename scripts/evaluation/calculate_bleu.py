import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

def calculate_bleu(ground_truth, model_output):
    """
    Calculate BLEU score between ground truth and model output texts
    
    reference: Ground truth text
    candidate: Generated text
    
    return: BLEU score
    """
    smooth_fn = SmoothingFunction().method3

    # Tokenize
    reference = [word_tokenize(ground_truth.lower(), language="slovene")]
    candidate = word_tokenize(model_output.lower(), language="slovene")
    
    # Calculate BLEU
    score = sentence_bleu(reference, candidate, weights = (0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)
    
    return score

def evaluate_json_file(json_file):
    """
    Evaluate BLEU scores for all entries in a JSON file
    
    json_file: Path to JSON file containing the data
    
    return: Dictionary with BLEU scores and evaluation details
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = [data]
    
    results = {
        'entries': [],
        'average_bleu': 0
    }
    
    total_score = 0
    valid_entries = 0
    
    for entry in data:
        if 'report' not in entry or 'output' not in entry:
            print(f"Warning: Entry missing 'report' or 'output' field, skipping")
            continue
            
        ground_truth = entry['report']
        candidate = entry['output']
        
        if not ground_truth or not candidate:
            print(f"Warning: Empty 'report' or 'output' field, skipping")
            continue
            
        score = calculate_bleu(ground_truth, candidate)
        total_score += score
        valid_entries += 1
        
        entry_result = {
            'date': entry.get('date', ''),
            'input': entry.get('input', ''),
            'bleu_score': score,
            'report': ground_truth[:100] + '...' if len(ground_truth) > 100 else ground_truth,
            'output': candidate[:100] + '...' if len(candidate) > 100 else candidate
        }
        results['entries'].append(entry_result)
    
    if valid_entries > 0:
        results['average_bleu'] = total_score / valid_entries
    
    return results

def print_results(results, file_name):
    """
    Print evaluation results
    """
    print(f"\nEvaluating file: {file_name.split('/')[-1]}")
    print(f"\nAverage BLEU score: {results['average_bleu']:.4f}")
    
    print("\nDetailed Results:")
    print("{:<25} {:<10}".format("Date", "BLEU"))
    print("-" * 35)
    
    for entry in results['entries']:
        print("{:<25} {:<10.4f}".format(
            entry['date'],
            entry['bleu_score']
        ))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate BLEU scores between ground truth and model output')
    parser.add_argument('--file', type=str, required=True, help='Path to JSON file containing the data')
    args = parser.parse_args()
    
    results = evaluate_json_file(args.file)
    print_results(results, args.file)