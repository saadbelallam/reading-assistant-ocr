#!/usr/bin/env python3
import os
import sys
import csv
import metrics_methods

def usage():
    print("Usage: metrics_evaluation.py <image_path>")
    print("  Example: metrics_evaluation.py tests/images/Documents_imprimes/easy/001.png")
    sys.exit(1)

def main():
    if len(sys.argv) != 2:
        usage()
    
    image_path = sys.argv[1]
    
    try:
        rel_path = os.path.relpath(image_path, "tests/images")
    except Exception as e:
        print(f"Error computing relative path: {e}")
        sys.exit(1)
    
    rel_no_ext = os.path.splitext(rel_path)[0]
    
    results_folder = os.path.join("tests/results", rel_no_ext)
    
    if not os.path.isdir(results_folder):
        print(f"Error: Results folder '{results_folder}' not found. Please process the image first.")
        sys.exit(1)
    
    ocr_file = os.path.join(results_folder, "6_nlp_output.txt")
    expected_file = results_folder.replace("results", "textes") + ".txt"
    
    if not os.path.isfile(ocr_file):
        print(f"Error: OCR output file '{ocr_file}' not found.")
        sys.exit(1)
    
    if not os.path.isfile(expected_file):
        print(f"Error: Expected file '{expected_file}' not found.")
        sys.exit(1)
    
    with open(ocr_file, 'r', encoding='utf-8') as f:
        ocr_output = f.read()
    
    with open(expected_file, 'r', encoding='utf-8') as f:
        original_input = f.read()

    # Calcul des tailles des textes
    original_text_length = len(original_input)
    ocr_text_length = len(ocr_output)
    
    metrics_computation = metrics_methods.MetricsComputation()
    metrics_computation.compute_all(ocr_output, original_input)
    
    print(metrics_computation.measures)
    
    measures_dict = {k: v for k, v in vars(metrics_computation.measures).items() 
                     if not isinstance(v, dict)}
    
    # Ajouter les tailles de texte au dictionnaire
    measures_dict["original_text_length"] = original_text_length
    measures_dict["ocr_text_length"] = ocr_text_length

    csv_path = os.path.join(results_folder, "metrics.csv")
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["image"] + list(measures_dict.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        row = {"image": rel_path}
        row.update(measures_dict)
        writer.writerow(row)

if __name__ == "__main__":
    main()
