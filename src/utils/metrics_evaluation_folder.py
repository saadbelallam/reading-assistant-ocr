#!/usr/bin/env python3
import os
import sys
import csv
import metrics_methods

def usage():
    print("Usage: metrics_evaluation_folder.py <images_folder>")
    print("  Example: ./metrics_evaluation_folder.py tests/images")
    sys.exit(1)

def is_image_file(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png'))

def main():
    if len(sys.argv) != 2:
        usage()

    images_folder = sys.argv[1]

    if not os.path.isdir(images_folder):
        print(f"Error: Folder '{images_folder}' not found.")
        sys.exit(1)

    print(f"Evaluating images in folder: {images_folder}")

    # CSV global
    csv_path = "tests/results/metrics.csv"
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = None

        for root, _, files in os.walk(images_folder):
            for file in files:
                if is_image_file(file):
                    image_path = os.path.join(root, file)
                    try:
                        rel_path = os.path.relpath(image_path, "tests/images")
                        rel_no_ext = os.path.splitext(rel_path)[0]
                        results_folder = os.path.join("tests/results", rel_no_ext)
                        expected_file = os.path.join("tests/textes", rel_no_ext + ".txt")
                        ocr_file = os.path.join(results_folder, "6_nlp_output.txt")

                        if not os.path.isdir(results_folder):
                            print(f"[SKIP] Missing results folder: {results_folder}")
                            continue
                        if not os.path.isfile(ocr_file):
                            print(f"[SKIP] Missing OCR file: {ocr_file}")
                            continue
                        if not os.path.isfile(expected_file):
                            print(f"[SKIP] Missing expected file: {expected_file}")
                            continue

                        with open(ocr_file, 'r', encoding='utf-8') as f:
                            ocr_output = f.read().strip()
                        with open(expected_file, 'r', encoding='utf-8') as f:
                            original_input = f.read().strip()

                        original_length = len(original_input)

                        metrics = metrics_methods.MetricsComputation()
                        metrics.compute_all(ocr_output, original_input)

                        print(f"\n=== {rel_path} ===")
                        for key, value in vars(metrics.measures).items():
                            print(f"{key}: {value}")
                        print(f"original_length: {original_length}")

                        # Construction du dictionnaire de résultats
                        measures_dict = {k: v for k, v in vars(metrics.measures).items()
                                         if not isinstance(v, dict)}
                        measures_dict["image"] = rel_path
                        measures_dict["original_length"] = original_length

                        if writer is None:
                            fieldnames = ["image", "original_length"] + list(measures_dict.keys())
                            fieldnames = list(dict.fromkeys(fieldnames))  # éviter doublons
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writeheader()

                        writer.writerow(measures_dict)

                    except Exception as e:
                        print(f"[ERROR] Processing {image_path}: {e}")

if __name__ == "__main__":
    main()
