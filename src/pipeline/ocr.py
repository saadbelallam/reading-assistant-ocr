#################
##    Step 5    #
#################

from pytesseract import Output
import pytesseract
import cv2
import numpy as np
from typing import List
from utils.draw_utils import draw_lines
def extract_text_from_region(preprocessed_image: np.ndarray, region: np.ndarray, masks: list, ocr_config: dict) -> str:
    """
    Process a single text line, masking unrecognized words while preserving line structure
    """
    # Crop region coordinates
    region = np.array(region)
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
        lang=ocr_config.get("lang", "fra"),
        config=f'--psm {ocr_config["psm"]} --oem {ocr_config["oem"]}',
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
            
        # Validate word: confidence > min_confidence and contains letters
        min_confidence = ocr_config.get("min_confidence", 40)
        if conf < min_confidence or not (any(c.isalpha() for c in text) or all(c in allowed_chars for c in text)):
            processed_line.append("<mask>")
            masks.append(text)
        else:
            processed_line.append(text)
        
    return ' '.join(processed_line)

def extract_text(image_path: str, output_dir: str, order: List, lines: List, lines_corr: dict, ocr_config: dict) -> str:
    print("Running OCR ...")
    masks = []
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    if order:
        ordered_lines = []
        for i in range(len(order)):
            idx = order[i]
            lines_indices = lines_corr[idx]
            lines_ = [lines[i] for i in lines_indices]
            sorted_lines = sorted(lines_, key=lambda box: box[:, 1].mean())
            ordered_lines+=sorted_lines

        # Draw lines for visualization
        draw_lines(image_path, ordered_lines,order, output_dir)
        # Batch processing to reduce per-call overhead
        full_text = []
        for region in ordered_lines:
            text = extract_text_from_region(image, region, masks, ocr_config)
            full_text.append(text)
    else:
        # Draw lines for visualization
        draw_lines(image_path,lines,order, output_dir)
        # Batch processing to reduce per-call overhead
        full_text = []
        for region in lines:
            text = extract_text_from_region(image, region, masks, ocr_config)
            full_text.append(text)

    # Save results
    text_output = '\n'.join(full_text)
    with open(f"{output_dir}/5_ocr_output.txt", "w") as f:
        f.write(text_output)
    return text_output, masks