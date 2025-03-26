#!/usr/bin/env python3
import os
import sys
import metrics_methods

def usage():
    print("Usage: metrics_evaluation.py <image_path>")
    print("  Example: results.py tests/images/Documents_imprimes/easy/001.png")
    sys.exit(1)

def main():
    if len(sys.argv) != 2:
        usage()
    
    # Input image path (e.g., tests/images/Documents_imprimes/easy/001.png)
    image_path = sys.argv[1]
    
    # Compute the relative path from tests/images
    try:
        rel_path = os.path.relpath(image_path, "tests/images")
    except Exception as e:
        print(f"Error computing relative path: {e}")
        sys.exit(1)
    
    # Remove the extension to get the folder name (e.g., "Documents_imprimes/easy/001")
    rel_no_ext = os.path.splitext(rel_path)[0]
    
    # Define the expected results folder path
    results_folder = os.path.join("tests/results", rel_no_ext)
    
    # Check if the results folder exists
    if not os.path.isdir(results_folder):
        print(f"Error: Results folder '{results_folder}' not found. Please process the image first.")
        sys.exit(1)
    
    # Define the OCR output and expected ground truth files
    ocr_file = os.path.join(results_folder, "5_ocr_output.txt")
    expected_file = os.path.join(results_folder, "expected.txt")
    
    if not os.path.isfile(ocr_file):
        print(f"Error: OCR output file '{ocr_file}' not found.")
        sys.exit(1)
    
    if not os.path.isfile(expected_file):
        print(f"Error: Expected file '{expected_file}' not found.")
        sys.exit(1)
    
    # Read the files
    with open(ocr_file, 'r', encoding='utf-8') as f:
        ocr_output = f.read()
    
    with open(expected_file, 'r', encoding='utf-8') as f:
        original_input = f.read()
    
    # Compute metrics using your metrics module
    metrics_computation = metrics_methods.MetricsComputation()
    metrics_computation.compute_all(ocr_output, original_input)
    
    # Print the evaluation results
    print(metrics_computation.measures)

if __name__ == "__main__":
    main()