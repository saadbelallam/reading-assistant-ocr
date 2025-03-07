#################
##    Step 5    #
#################

from typing import List
import cv2
import numpy as np
import pytesseract

def extract_text_from_region(preprocessed_image: np.ndarray, region: np.ndarray) -> str:
    """
    Extract text from a region with minimal overhead.
    """
    x_coords = region[:, 0]
    y_coords = region[:, 1]
    x1, x2 = int(np.min(x_coords)), int(np.max(x_coords))
    y1, y2 = int(np.min(y_coords)), int(np.max(y_coords))
    cropped = preprocessed_image[y1:y2, x1:x2]
    
    if cropped.size == 0:
        return ""
    
    return pytesseract.image_to_string(
        cropped, 
        lang='fra', 
        config='--psm 7 --oem 3'
    ).strip()

def extract_text(image_path: str, output_dir: str, ordered_lines: List[np.ndarray]) -> str:
    print("Running OCR ...")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Preprocess ONCE for all regions
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, preprocessed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Batch processing to reduce per-call overhead
    full_text = []
    for region in ordered_lines:
        text = extract_text_from_region(preprocessed, region)
        full_text.append(text)

    # Save results
    text_output = '\n'.join(full_text)
    with open(f"{output_dir}/5_ocr_output.txt", "w") as f:
        f.write(text_output)
    return text_output