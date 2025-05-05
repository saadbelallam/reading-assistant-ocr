################
#    Step 3    #
################

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional

def order_blocks(
    image_path: str,  
    blocks: List[np.ndarray],  
    output_dir: str = "tests/results"
) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """
    Orders text blocks using an iterative X-Y Cut algorithm.
    """
    print("Running blocks ordering XY ...")
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Convert contours to bounding boxes (x, y, w, h)
    boxes = [(block[0][0], block[0][1], block[2][0] - block[0][0], block[2][1] - block[0][1]) for block in blocks]

    # Order the blocks using X-Y cut
    ordered_boxes, order = xy_cut_sort(image, boxes)

    # Generate visualization
    vis_image = visualize_xycut_order(image.copy(), ordered_boxes)

    # Save results
    cv2.imwrite(f"{output_dir}/3_ordered_blocks.jpg", vis_image)

    return ordered_boxes, order


def xy_cut_sort(image: np.ndarray, boxes: List[Tuple[int, int, int, int]], min_gap: int = 20) -> List[Tuple[int, int, int, int]]:
    """
    Iterative X-Y Cut sorting using a queue.
    """
    rects = np.array([(x, y, x + w, y + h) for (x, y, w, h) in boxes])
    queue = [(rects, image.shape, True)]  # (bounding_boxes, shape, vertical_first)
    sorted_rects = []
    splits = 0
    MAX_SPLITS = 20  # Prevent infinite loop

    while queue:
        rects, shape, vertical_first = queue.pop(0)
        if len(rects) == 0:
            continue

        if splits >= MAX_SPLITS:  # Stop excessive splitting
            sorted_rects.extend(sorted(rects, key=lambda r: (r[1], r[0])))
            continue
        
        splits += 1  # Track splits

        # Projection-based splitting
        projection = np.zeros(shape[1 if vertical_first else 0], dtype=int)
        for x1, y1, x2, y2 in rects:
            projection[x1:x2] += 1 if vertical_first else projection[y1:y2] + 1

        cut_pos = _find_cut_position(projection, min_gap)
        if cut_pos is None:
            sorted_rects.extend(sorted(rects, key=lambda r: (r[1], r[0])))
            continue  # Stop adding to queue if no valid cut is found

        # Split rectangles
        if vertical_first:
            left = rects[rects[:, 2] <= cut_pos]
            right = rects[rects[:, 0] > cut_pos]
            if len(left) == len(rects) or len(right) == len(rects):  # Prevent adding the same group
                sorted_rects.extend(sorted(rects, key=lambda r: (r[1], r[0])))
                continue
            queue.extend([(left, shape, True), (right, shape, True)])
        else:
            top = rects[rects[:, 3] <= cut_pos]
            bottom = rects[rects[:, 1] > cut_pos]
            if len(top) == len(rects) or len(bottom) == len(rects):  # Prevent infinite loop
                sorted_rects.extend(sorted(rects, key=lambda r: (r[1], r[0])))
                continue
            queue.extend([(top, shape, True), (bottom, shape, True)])

    return [(x, y, x2 - x, y2 - y) for (x, y, x2, y2) in sorted_rects]


def _find_cut_position(projection: np.ndarray, min_gap: int) -> Optional[int]:
    """
    Finds the largest gap in the projection profile that exceeds min_gap.
    Returns the split position or None if no suitable gap is found.
    """
    # Get indices where projection is zero (whitespace)
    zero_regions = np.where(projection == 0)[0]

    if len(zero_regions) == 0:
        return None  # No gaps found

    # Find where the gaps between zero-regions exceed 1 pixel
    diff = np.diff(zero_regions)
    gaps = np.where(diff > 1)[0]

    if len(gaps) == 0:
        return None  # No wide enough gaps found

    # Get the largest gap
    max_gap_idx = np.argmax(diff[gaps]) if len(gaps) > 0 else None

    if max_gap_idx is None:
        return None  # No valid cut position

    cut_pos = zero_regions[gaps[max_gap_idx]]

    # Ensure the gap is large enough before returning
    return cut_pos if diff[gaps[max_gap_idx]] >= min_gap else None


def visualize_xycut_order(
    image: np.ndarray,
    ordered_boxes: List[Tuple[int, int, int, int]]
) -> np.ndarray:
    """
    Draws X-Y Cut ordered blocks with numbering, where number size scales with image dimensions.
    """
    height, width = image.shape[:2]

    # Scale font size and thickness based on image size
    base_size = min(width, height)  # Take smaller dimension as base reference
    font_scale = max(0.5, base_size / 1000)  # Ensure it’s not too small
    thickness = max(1, int(base_size / 500))  # Minimum thickness of 1

    for idx, (x, y, w, h) in enumerate(ordered_boxes, 1):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), thickness)
        cv2.putText(
            image, str(idx), (x + 5, y + int(h / 2)),  
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness
        )

    return image

