################
#    Step 4    #
################

from models.craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)
import numpy as np

def detect_lines(image_path, output_dir, ordered_blocks):
    print("Running lines detection ...")
    image = read_image(image_path)
    craft_net = load_craftnet_model(cuda=False)
    refine_net = load_refinenet_model(cuda=False)

    # Detect lines in the FULL IMAGE
    prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.7,
        link_threshold=0.7,
        low_text=0.4,
        cuda=False,
        long_size=1280,
    )
    all_boxes = prediction_result["boxes"]
    
    # Group lines by their corresponding blocks
    block_lines = [[] for _ in ordered_blocks]

    for box in all_boxes:
        # Calculate the center of the detected line
        center_x = np.mean(box[:, 0])
        center_y = np.mean(box[:, 1])

        # Assign line to the first matching block (assuming non-overlapping blocks)
        for i, (x, y, w, h) in enumerate(ordered_blocks):
            if x <= center_x <= x + w and y <= center_y <= y + h:
                block_lines[i].append(box)
                break

    # Sort lines within each block by y-coordinate (top to bottom)
    sorted_lines = []
    for i, block in enumerate(ordered_blocks):
        lines_in_block = block_lines[i]
        lines_in_block.sort(key=lambda b: np.min(b[:, 1]))  # Sort by topmost y-coordinate
        sorted_lines.extend(lines_in_block)

    # Export all lines in reading order
    export_detected_regions(image=image, regions=sorted_lines, rectify=True)
    export_extra_results(image=image, regions=sorted_lines, output_path=output_dir)
    empty_cuda_cache()

    return sorted_lines