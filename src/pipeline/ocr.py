#################
##    Step 5    #
#################
#
#import os
#import sys
#import pytesseract
#import cv2
#import numpy as np
#import re
#from typing import List, Tuple
#
#def extract_text_from_block(block: np.ndarray) -> str:
#    """
#    Extract text from a single block using OCR.
#    
#    :param block: A cropped image block containing text
#    :return: Extracted text
#    """
#    if block is None or not isinstance(block, np.ndarray):
#        print("Error: Invalid block encountered")
#        return ""
#    
#    # Convert the block to grayscale (if not already)
#    gray_block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
#    
#    # for debugging
#    #cv2.imshow("Gray Block", gray_block)
#    #cv2.waitKey(0)  # Wait for a key press to close the window
#    #cv2.destroyAllWindows()
#    
#    # Use pytesseract to extract text
#    text = pytesseract.image_to_string(gray_block, lang='eng', config=r' --psm 7')
#    #print(text.strip())
#    return text.strip()
#
#
#def natural_sort_key(s):
#    """Sort file names numerically (e.g., crop_1.png before crop_10.png)."""
#    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]
#
#
## Function to process all blocks
#def extract_text(input_dir: str, output_file: str) -> str:
#    """
#    Process all `crop_*.png` images in `input_dir` and extract text.
#    
#    :param input_dir: Directory containing cropped images
#    :param output_file: Path to save the extracted text
#    """
#    print("Running OCR on cropped images...")
#    
#    input_dir += "/image_crops"
#    if not os.path.exists(input_dir):
#        raise FileNotFoundError(f"Input directory not found: {input_dir}")
#    
#    #crop_files = sorted(f for f in os.listdir(input_dir) if f.startswith("crop_") and f.endswith(".png"), key=natural_sort_key)
#    crop_files = sorted(
#        (f for f in os.listdir(input_dir) if f.startswith("crop_") and f.endswith(".png")),
#        key=natural_sort_key
#    )
#    if not crop_files:
#        print("No crop_*.png files found in the directory.")
#        return
#    
#    extracted_texts = ""
#    
#    for crop_file in crop_files:
#        image_path = os.path.join(input_dir, crop_file)
#        image = cv2.imread(image_path)
#        
#        if image is None:
#            print(f"Warning: Unable to read {crop_file}")
#            continue
#        
#        text = extract_text_from_block(image)
#        #extracted_texts.append(f"{crop_file}:\n{text}\n")
#        extracted_texts += text + "\n"
#        
#    text_output_path = os.path.join(output_file, "5_ocr_output.txt")
#    with open(text_output_path, "w", encoding="utf-8") as f:
#        f.write(extracted_texts)
#    
#    return extracted_texts

################
#    Step 5    #
################

import os
import pytesseract
import cv2
import numpy as np
from typing import List

def extract_text_from_region(image: np.ndarray, region: np.ndarray) -> str:
    """
    Extract text from a specific region of the image using OCR.
    
    :param image: Full original image
    :param region: Numpy array of shape (4, 2) representing quadrilateral coordinates
    :return: Extracted text
    """
    # Convert quadrilateral to rectangle
    x_coords = region[:, 0]
    y_coords = region[:, 1]
    x1, x2 = int(np.min(x_coords)), int(np.max(x_coords))
    y1, y2 = int(np.min(y_coords)), int(np.max(y_coords))

    # Crop and process the region
    cropped = image[y1:y2, x1:x2]
    
    # Skip empty regions
    if cropped.size == 0:
        return ""

    # Preprocess for OCR
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # OCR with configuration for single line text
    return pytesseract.image_to_string(
        processed, 
        lang='fra',
        config='--psm 7 --oem 3'  # Treat image as single text line
    ).strip()

def extract_text(
    image_path: str, 
    output_dir: str,
    ordered_lines: List[np.ndarray]
) -> str:
    """
    Process text regions in reading order and extract text
    
    :param image_path: Path to original document image
    :param output_dir: Directory to save OCR results
    :param ordered_lines: List of regions in reading order from previous step
    :return: Concatenated extracted text
    """
    print("Running OCR ...")
    
    # Load original image once
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    full_text = []
    
    # Process each region in order
    for idx, region in enumerate(ordered_lines, 1):
        text = extract_text_from_region(image, region)
        full_text.append(text)
        
        # Debugging: Save cropped region for verification
        # region_image = image[
        #     int(np.min(region[:,1])):int(np.max(region[:,1])),
        #     int(np.min(region[:,0])):int(np.max(region[:,0]))
        # ]
        # cv2.imwrite(f"{output_dir}/line_{idx:04d}.png", region_image)

    # Save final text
    text_output = '\n'.join(full_text)
    with open(f"{output_dir}/5_ocr_output.txt", "w") as f:
        f.write(text_output)
        
    return text_output