#!/usr/bin/env python3
import os
import sys
import metrics_methods

def usage():
    print("Usage: metrics_evaluation.py <input_directory>")
    print("  Example: ./results.py tests/images/Documents_imprimes/easy/")
    sys.exit(1)

def process_image(image_path):
    try:
        rel_path = os.path.relpath(image_path, "tests/images")
        rel_no_ext = os.path.splitext(rel_path)[0]
        results_folder = os.path.join("tests/results", rel_no_ext)
        
        if not os.path.isdir(results_folder):
            print(f"Skipping: Results folder '{results_folder}' not found")
            return None

        ocr_file = os.path.join(results_folder, "5_ocr_output.txt")
        expected_file = os.path.join(results_folder, "expected.txt")

        if not all(os.path.isfile(f) for f in [ocr_file, expected_file]):
            print(f"Skipping: Missing files in {results_folder}")
            return None

        with open(ocr_file, 'r', encoding='utf-8') as f:
            ocr_output = f.read().strip()
        with open(expected_file, 'r', encoding='utf-8') as f:
            original_input = f.read().strip()

        # Calcul des caractéristiques du texte
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
        print(f"Error processing {image_path}: {str(e)}")
        return None

def main():
    if len(sys.argv) != 2:
        usage()
    
    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found")
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

    total_files = 0
    success_count = 0

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                total_files += 1
                image_path = os.path.join(root, file)
                result = process_image(image_path)
                
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

    # Rapport final
    print("\n=== Rapport pondéré par longueur de texte ===")
    print(f"Nombre total de caractères analysés: {absolute_metrics['total_chars']}")
    print(f"Images traitées: {success_count}/{total_files}")
    
    print("\nMétriques pondérées:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nMétriques globales:")
    print(f"Précision caractère globale: {global_character_accuracy:.2f}%")
    print(f"Précision mot globale: {global_precision:.4f}")
    print(f"Rappel mot global: {global_recall:.4f}")
    print(f"F1-score mot global: {global_f1:.4f}")
    
    print("\nErreurs absolues (total):")
    print(f"Lev Distance: {absolute_metrics['levenshtein_distance']}")
    print(f"Substitutions: {absolute_metrics['substitutions']}")
    print(f"Insertions: {absolute_metrics['insertions']}")
    print(f"Deletions: {absolute_metrics['deletions']}")
    
    print("\nErreurs relatives (pour 1000 caractères):")
    if absolute_metrics['total_chars'] > 0:
        chars_per_1000 = absolute_metrics['total_chars'] / 1000
        print(f"Lev Distance: {absolute_metrics['levenshtein_distance'] / chars_per_1000:.1f}")
        print(f"Substitutions: {absolute_metrics['substitutions'] / chars_per_1000:.1f}")
        print(f"Insertions: {absolute_metrics['insertions'] / chars_per_1000:.1f}")
        print(f"Deletions: {absolute_metrics['deletions'] / chars_per_1000:.1f}")

if __name__ == "__main__":
    main()