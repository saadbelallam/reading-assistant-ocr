################
#    Step 2    #
################

#!/usr/bin/env python3
# 2_blocks_detection.py

import cv2
import numpy as np
import os
import argparse

def detect_text_blocks_rlsa(image_path: str, output_dir: str = "tests/results/") -> list:
    """
    Detect text blocks using morphological closing (RLSA-inspired method) with adaptive kernel sizing.

    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save visualization image.

    Returns:
        list: List of detected block contours.
    """
    print("Running texts region detection ...")
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Load image and create a copy for visualization
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or cannot be loaded!")
        return []
    orig = image.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize the image using Otsu's threshold (invert so text is white)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Get image dimensions
    height, width = gray.shape[:2]

    # Adaptive kernel sizes and minimum area based on image size
    vertical_kernel_size = max(1, height // 50)  # ~2% of image height
    horizontal_kernel_size = max(1, width // 50)  # ~2% of image width
    min_area = (height * width) // 500  # ~0.2% of total image area

    # print(f"Using kernel sizes: Vertical={vertical_kernel_size}, Horizontal={horizontal_kernel_size}")
    # print(f"Minimum area threshold: {min_area}")

    # Apply vertical closing to merge lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_size))
    closed_vertical = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)

    # Apply horizontal closing to merge words into blocks
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_size, 1))
    closed = cv2.morphologyEx(closed_vertical, cv2.MORPH_CLOSE, horizontal_kernel, iterations=1)

    # Dilate slightly to smooth regions
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.dilate(closed, dilation_kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area and return them
    detected_blocks = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    # Generate visualization
    vis = orig.copy()
    for cnt in detected_blocks:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save visualization with dynamic naming
    vis_path = os.path.join(output_dir, f"2_detected_blocks.png")
    cv2.imwrite(vis_path, vis)
    #print(f"Detected {len(detected_blocks)} text blocks. Visualization saved to '{vis_path}'.")

    return detected_blocks

#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description="Detect text blocks using an adaptive RLSA-inspired method.")
#    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
#    parser.add_argument("--output", type=str, default="tests/results/", help="Directory to save results.")
#
#    args = parser.parse_args()
#    blocks = detect_text_blocks_rlsa(args.image, args.output)