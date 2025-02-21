# pytesseract vs. easyocr vs keras_ocr
# modulable

################
#    Step 5    #
################

import os
import sys
import pytesseract
import cv2
import numpy as np
from typing import List, Tuple

def extract_text_from_block(block: np.ndarray) -> str:
    """
    Extract text from a single block using OCR.
    
    :param block: A cropped image block containing text
    :return: Extracted text
    """
    if block is None or not isinstance(block, np.ndarray):
        print("Error: Invalid block encountered")
        return ""
    
    # Convert the block to grayscale (if not already)
    gray_block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
    
    # for debugging
    #cv2.imshow("Gray Block", gray_block)
    #cv2.waitKey(0)  # Wait for a key press to close the window
    #cv2.destroyAllWindows()
    
    # Use pytesseract to extract text
    text = pytesseract.image_to_string(gray_block, lang='fra')
    #print(text.strip())
    return text.strip()


# Function to process all blocks
def extract_text(image_path: str, ordered_blocks: List[Tuple[int, int, int, int]], output_dir: str = "tests/results") -> str:
    """
    Process each block and extract text.
    
    :param ordered_blocks: List of ordered image blocks
    :return: Full extracted text as a single string
    """
    print("Running OCR ...")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    # output_dir = "tests/ordered_blocks" # saving for debuging
    # os.makedirs(output_dir, exist_ok=True) # saving for debuging
    
    full_text = []
    
    for idx, (x, y, w, h) in enumerate(ordered_blocks):
        block = image[y:y+h, x:x+w]
        save_path = os.path.join(output_dir, f"ordered_block_{idx}.png") # saving for debuging
        cv2.imwrite(save_path, block) # saving for debuging
        block_text = extract_text_from_block(block)
        full_text.append(block_text)
    
    full_text_str = '\n'.join(full_text)
    
    text_output_path = os.path.join(output_dir, "5_ocr_output.txt")
    with open(text_output_path, "w", encoding="utf-8") as f:
        f.write(full_text_str)
    
    return full_text_str