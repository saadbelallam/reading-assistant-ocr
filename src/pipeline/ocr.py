#################
##    Step 5    #
#################

from pytesseract import Output
import pytesseract
import cv2
import numpy as np
from typing import List

def extract_text_from_region(preprocessed_image: np.ndarray, region: np.ndarray) -> str:
    """
    Process a single text line, masking unrecognized words while preserving line structure
    """
    # Crop region coordinates
    x_coords = region[:, 0]
    y_coords = region[:, 1]
    x1, x2 = int(np.min(x_coords)), int(np.max(x_coords))
    y1, y2 = int(np.min(y_coords)), int(np.max(y_coords))
    cropped = preprocessed_image[y1:y2, x1:x2]
    
    if cropped.size == 0:
        return ""
    
    # Get OCR data with word-level information
    data = pytesseract.image_to_data(
        cropped,
        lang='fra',
        config='--psm 7 --oem 1',  # Treat as single text line
        output_type=Output.DICT
    )
    
    # Caractères spéciaux et nombres autorisés
    allowed_chars = {':', '-', '.', ',', ';', '!', '?', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

    processed_line = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = float(data['conf'][i])
        
        # Skip empty strings and non-text elements
        if not text or conf < 0:
            continue
            
        # Validate word: confidence > 60 and contains letters
        if conf < 60 or not (any(c.isalpha() for c in text) or all(c in allowed_chars for c in text)):
            processed_line.append("<mask>")
        else:
            processed_line.append(text)
        
    return ' '.join(processed_line)

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