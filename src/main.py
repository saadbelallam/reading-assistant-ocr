#!/usr/bin/env python3
import argparse
import os
import time
import shutil

from pipeline.geometric_correction import scan_document
from pipeline.blocks_detection import detect_text_blocks_rlsa
from pipeline.blocks_ordering import order_blocks
from pipeline.lines_detection import detect_lines
from pipeline.ocr import extract_text
from pipeline.nlp import correct_text
from pipeline.audio_generation import generate

from utils.timing_out import timing_out

def main():
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
        print("===============> No corresponding text file found for input image")
    
    # Initialize timing report dictionary
    timing_report = {}
    
    # Start total processing time
    start_time = time.time()

    # 1. Geometric correction could go here if needed.
    start_geo = time.time()
    image_path = scan_document(image_path, result_path)
    timing_report["Geometric Correction"] = time.time() - start_geo
        
    # 2. Block detection – store intermediate results in result_path
    start_block_detect = time.time()
    blocks = detect_text_blocks_rlsa(image_path, result_path)
    timing_report["Block Detection"] = time.time() - start_block_detect

    # 3. Blocks ordering – also store ordering visualization in result_path
    start_block_order = time.time()
    ordered_blocks, _ = order_blocks(image_path, blocks, result_path)
    timing_report["Block Ordering"] = time.time() - start_block_order
    
    # 4. Lines detection using Craft
    start_lines_detect = time.time()
    lines = detect_lines(image_path, result_path, ordered_blocks)
    timing_report["Lines Detection"] = time.time() - start_lines_detect

    # 5. OCR processing – extract text using ordered lines
    start_ocr = time.time()
    text = extract_text(image_path, result_path, lines)
    timing_report["OCR Processing"] = time.time() - start_ocr

    # 6. NLP text correction – correct the OCR text
    start_nlp = time.time()
    text = correct_text(text, result_path)
    timing_report["NLP Correction"] = time.time() - start_nlp

    # 7. Audio generation
    # start_audio = time.time()
    # generate(text, result_path)
    # timing_report["Audio generation"] = time.time() - start_audio
    
    # Timing report printing
    total_time = time.time() - start_time
    timing_report["Total Processing Time"] = total_time
    
    if show_timing:
        timing_out(timing_report)

if __name__ == "__main__":
    main()