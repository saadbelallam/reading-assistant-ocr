#!/usr/bin/env python3
import os
import sys
import metrics_methods

def usage():
    print("Usage: metrics_evaluation_folder.py <results_directory>")
    print("  Example: ./metrics_evaluation_folder.py tests/results/")
    sys.exit(1)

def process_result_dir(result_dir):
    try:
        ocr_file = os.path.join(result_dir, "5_ocr_output.txt")
        expected_file = os.path.join(result_dir, "expected.txt")

        if not all(os.path.isfile(f) for f in [ocr_file, expected_file]):
            print(f"Skipping: Missing files in {result_dir}")
            return None

        with open(ocr_file, 'r', encoding='utf-8') as f:
            ocr_output = f.read().strip()
        with open(expected_file, 'r', encoding='utf-8') as f:
            original_input = f.read().strip()

        original_char_count = len(original_input)
        original_word_count = len(original_input.split())

        metrics = metrics_methods.MetricsComputation()
        metrics.compute_all(ocr_output, original_input)
        
        return {
            'metrics': {
                'levenshtein_distance': metrics.measures.levenshetein_distance,
                'character_accuracy': metrics.measures.character_accuracy,
                'precision': metrics.measures.precision,
                'recall': metrics.measures.recall,
                'F1_score': metrics.measures.F1_score,
                'substitutions': metrics.measures.substitutions,
                'deletions': metrics.measures.deletions,
                'insertions': metrics.measures.insertions
            },
            'counts': {
                'original_chars': original_char_count,
                'original_words': original_word_count,
                'TP': metrics.measures.TP,
                'FP': metrics.measures.FP,
                'FN': metrics.measures.FN
            }
        }

    except Exception as e:
        print(f"Error processing {result_dir}: {str(e)}")
        return None

def main():
    if len(sys.argv) != 2:
        usage()
    
    results_root = sys.argv[1]
    if not os.path.isdir(results_root):
        print(f"Error: Results directory '{results_root}' not found")
        sys.exit(1)

    total_weight = 0
    weighted_metrics = {
        'character_accuracy': 0,
        'precision': 0,
        'recall': 0,
        'F1_score': 0
    }
    
    absolute_metrics = {
        'levenshtein_distance': 0,
        'substitutions': 0,
        'deletions': 0,
        'insertions': 0,
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'total_chars': 0
    }

    total_dirs = 0
    success_count = 0

    # Walk through all result directories
    for root, dirs, files in os.walk(results_root):
        # Skip directories that don't contain both required files
        if ('5_ocr_output.txt' in files) and ('expected.txt' in files):
            total_dirs += 1
            result_dir = root
            result = process_result_dir(result_dir)
            
            if result:
                success_count += 1
                weight = result['counts']['original_chars']
                total_weight += weight
                
                for metric in weighted_metrics:
                    weighted_metrics[metric] += result['metrics'][metric] * weight
                
                for metric in absolute_metrics:
                    if metric in result['counts']:
                        absolute_metrics[metric] += result['counts'][metric]
                    elif metric in result['metrics']:
                        absolute_metrics[metric] += result['metrics'][metric]
                
                absolute_metrics['total_chars'] += weight

    final_metrics = {}
    if total_weight > 0:
        for metric, value in weighted_metrics.items():
            final_metrics[metric] = value / total_weight
    else:
        for metric in weighted_metrics:
            final_metrics[metric] = 0

    try:
        global_character_accuracy = (1 - absolute_metrics['levenshtein_distance'] / absolute_metrics['total_chars']) * 100
    except ZeroDivisionError:
        global_character_accuracy = 0

    try:
        global_precision = absolute_metrics['TP'] / (absolute_metrics['TP'] + absolute_metrics['FP'])
    except ZeroDivisionError:
        global_precision = 0
        
    try:
        global_recall = absolute_metrics['TP'] / (absolute_metrics['TP'] + absolute_metrics['FN'])
    except ZeroDivisionError:
        global_recall = 0
        
    try:
        global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall)
    except ZeroDivisionError:
        global_f1 = 0

    print("\n=== Weighted Report by Text Length ===")
    print(f"Total analyzed characters: {absolute_metrics['total_chars']}")
    print(f"Processed results: {success_count}/{total_dirs}")
    
    print("\nWeighted Metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nGlobal Metrics:")
    print(f"Character Accuracy: {global_character_accuracy:.2f}%")
    print(f"Word Precision: {global_precision:.4f}")
    print(f"Word Recall: {global_recall:.4f}")
    print(f"Word F1-Score: {global_f1:.4f}")
    
    print("\nAbsolute Errors (total):")
    print(f"Lev Distance: {absolute_metrics['levenshtein_distance']}")
    print(f"Substitutions: {absolute_metrics['substitutions']}")
    print(f"Insertions: {absolute_metrics['insertions']}")
    print(f"Deletions: {absolute_metrics['deletions']}")
    
    print("\nRelative Errors (per 1000 characters):")
    if absolute_metrics['total_chars'] > 0:
        chars_per_1000 = absolute_metrics['total_chars'] / 1000
        print(f"Lev Distance: {absolute_metrics['levenshtein_distance'] / chars_per_1000:.1f}")
        print(f"Substitutions: {absolute_metrics['substitutions'] / chars_per_1000:.1f}")
        print(f"Insertions: {absolute_metrics['insertions'] / chars_per_1000:.1f}")
        print(f"Deletions: {absolute_metrics['deletions'] / chars_per_1000:.1f}")

if __name__ == "__main__":
    main()