#################
##    Step 5    #
#################

from pytesseract import Output
import pytesseract
import easyocr
import cv2
import numpy as np
from typing import List

# Cache EasyOCR pour ne pas recharger les poids à chaque appel ---------------
_easy_reader_cache = {}

def _get_easy_reader(lang: str = "fr", gpu: bool = False):
    """
    récupère un reader EasyOCR
    """
    lang_key = tuple(lang)
    lang_arg = list(lang)
    _easy_reader_cache[lang_key] = easyocr.Reader(lang_arg, gpu=gpu)
    return _easy_reader_cache[lang_key]



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
    
    # Caractères spéciaux et nombres autorisés
    allowed_chars = {':', '-', '.', ',', ';', '!', '?', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

    processed_line = []

    # pytesseract ------------------------------------------------------
    if ocr_config["ocr"] == "pytesseract":
        print("utilisationn de pytesseract")
        # Get OCR data with word-level information
        data = pytesseract.image_to_data(
            cropped,
            lang=ocr_config.get("lang", "fra"),
            config=f'--psm {ocr_config["psm"]} --oem {ocr_config["oem"]}',
            output_type=Output.DICT,
        )
    
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
    # EASYOCR -----------------------------------------------------------
    elif ocr_config['ocr'] == "easyocr":
        print("utilisation de easyocr")
        reader = _get_easy_reader(['fr'], gpu=ocr_config.get("gpu", False))
        # detail=1 → [(bbox, text, conf), ...]
        for _, text, conf in reader.readtext(cropped, detail=1, paragraph=False):
            text = text.strip()
            # Skip empty strings and non-text elements
            if not text:
                continue
            # Validate word: confidence > min_confidence and contains letters
            min_confidence = ocr_config.get("min_confidence", 40)
            if conf*100 < min_confidence or not (any(c.isalpha() for c in text) or all(c in allowed_chars for c in text)):
                processed_line.append("<mask>")
                masks.append(text)
            else:
                processed_line.append(text)

    else:
        raise ValueError("ocr_config['ocr'] doit être 'pytesseract' ou 'easyocr'")
    return ' '.join(processed_line)

def extract_text(image_path: str, output_dir: str, order: List, lines: List, lines_corr: dict, ocr_config: dict) -> str:
    print("Running OCR ...")
    masks = []
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    ordered_lines = []
    for i in range(len(order)):
        idx = order[i]
        lines_indices = lines_corr[idx]
        lines_ = [lines[i] for i in lines_indices]
        sorted_lines = sorted(lines_, key=lambda box: box[:, 1].mean())
        ordered_lines+=sorted_lines
    print(len(ordered_lines))

    # Batch processing to reduce per-call overhead
    full_text = []
    for region in ordered_lines:
        text = extract_text_from_region(image, region, masks, ocr_config)
        full_text.append(text)

    # Save results
    text_output = '\n'.join(full_text)
    with open(f"{output_dir}/5_ocr_output.txt", "w") as f:
        f.write(text_output)
    return text_output, masks