#!/usr/bin/env python3
import argparse
import os
import time
import shutil
import json
import psutil
import csv

from pipeline.geometric_correction import geometric_correction
from pipeline.blocks_detection import detect_text_blocks_stat
from pipeline.blocks_detection import detect_text_blocks_hdbscan
from pipeline.blocks_ordering_XY import order_blocks
from pipeline.heatmap_ordering import get_ordered_polygons
from pipeline.blocks_ordering_ML import reading_order_blocks
from pipeline.lines_detection import detect_lines
from pipeline.ocr import extract_text
from pipeline.nlp import correct_text
from pipeline.audio_generation import generate

from utils.timing_out import timing_out

def load_config():
    """Load configuration from settings.json"""
    with open("config/settings.json", "r") as f:
        return json.load(f)
    
def main():
    config = load_config()

    parser = argparse.ArgumentParser(description="Process a document image through the pipeline.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output-dir", type=str, required=True, help="Base output directory for results")
    parser.add_argument("-t", "--timing", action="store_true", help="Show execution time for each step")
    args = parser.parse_args()

    image_path = args.input
    output_base = args.output_dir
    show_timing = args.timing

    os.makedirs(output_base, exist_ok=True)
    
    rel_path = os.path.relpath(image_path, start="tests/images")  # Remove leading `tests/images`
    rel_path = os.path.splitext(rel_path)[0]  # Remove file extension
    result_path = os.path.join(output_base, rel_path)  # Final result path
    os.makedirs(result_path, exist_ok=True)
    
    print(f"Results will be stored in: {result_path}")
    
    # Save the expectation in the result fold
    base_input, _ = os.path.splitext(image_path)
    corresponding_txt = base_input + ".txt"
    if os.path.exists(corresponding_txt):
        print("Copying the expected result ...")
        dest_txt = os.path.join(result_path, "expected.txt")
        shutil.copy(corresponding_txt, dest_txt)
    else:
        print("└──No corresponding text file found for input image")
    
    # Initialize timing report dictionary
    timing_report = {}
    
    # Start total processing time
    start_time = time.time()

    try:
        # 1. Geometric correction
        if config["geometric_correction"].get("enabled", False):
            start_geo = time.time()
            image_path = geometric_correction(image_path, result_path)
            timing_report["Geometric Correction"] = time.time() - start_geo

        # 2. Lines detection using Craft
        if config["lines_detection"].get("enabled", False):
            start_lines_detect = time.time()
            lines = detect_lines(image_path, result_path)
            timing_report["Lines Detection"] = time.time() - start_lines_detect

        # si blocks_detection est désactivé on crée un mapping 1‑à‑1
        if not config["blocks_detection"].get("enabled", False):
            lines_corr = {idx: [idx] for idx in range(len(lines))}

        # 3. Block detection – store intermediate results in result_path
        if config["blocks_detection"].get("enabled", False):
            start_block_detect = time.time()
            if config["blocks_detection"].get("strategy", "STAT") == "STAT":
                blocks, lines_corr = detect_text_blocks_stat(image_path, result_path, lines, visualize=True)
            elif config["blocks_detection"].get("strategy", "STAT") == "HDBSCAN":
                blocks, lines_corr = detect_text_blocks_hdbscan(image_path, result_path, lines, visualize=True)
                print(blocks)
            timing_report["Block Detection"] = time.time() - start_block_detect


        # si blocks_ordering est désactivé 
        if not config["blocks_detection"].get("enabled", False):
            blocks = lines
            order = []
        # 4. Blocks ordering
        if config["blocks_ordering"].get("enabled", False):
            start_block_order = time.time()
            if config["blocks_ordering"].get("startegy", "ML") == "ML":
                ordered_blocks, order = reading_order_blocks(image_path, blocks, result_path)
            elif config["blocks_ordering"].get("startegy", "ML") == "HM":
                ordered_blocks, order = get_ordered_polygons(image_path, blocks, result_path)
            else:
                ordered_blocks, order = order_blocks(image_path, blocks, result_path)
            timing_report["Block Ordering"] = time.time() - start_block_order

        # 5. OCR processing – extract text using ordered lines
        if config["ocr"].get("enabled", False):
            start_ocr = time.time()
            text, masks = extract_text(image_path, result_path, order, lines, lines_corr, config["ocr"])
            timing_report["OCR Processing"] = time.time() - start_ocr

        # 6. NLP text correction – correct the OCR text
        if config["nlp"].get("enabled", False):
            start_nlp = time.time()
            text = correct_text(text, masks, result_path, config["nlp"])
            timing_report["NLP Correction"] = time.time() - start_nlp

        # 7. Audio generation
        if config["audio_generation"].get("enabled", False):
            start_audio = time.time()
            generate(text, result_path)
            timing_report["Audio generation"] = time.time() - start_audio

    except KeyError as e:
        print(f"Missing configuration key: {e}")
        return
    
    # Timing report printing
    total_time = time.time() - start_time
    timing_report["Total Processing Time"] = total_time
    # RAM
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info().rss / (1024 * 1024) 

    # ✅ Save total time and RAM usage to a CSV file
    csv_path = os.path.join(output_base, "time_ram.csv")
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Image", "Total Time (s)", "RAM (MB)"])
        image_name = os.path.basename(image_path)
        writer.writerow([image_name, f"{total_time:.2f}", f"{mem_info:.2f}"])

    if show_timing:
        timing_out(timing_report)

if __name__ == "__main__":
    main()