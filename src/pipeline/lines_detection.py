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
    """
    Detect text lines in each ordered block using the CRAFT text detector.

    Parameters:
        image_path (str): Path to the input image.
        output_dir (str): Directory where exported results will be saved.
        ordered_blocks (List[Tuple[int, int, int, int]]): Ordered list of blocks (x, y, w, h).

    Returns:
        list: Ordered list of detected lines (boxes) in reading order.
    """
    print("Running lines detection ...")
    image = read_image(image_path)
    craft_net = load_craftnet_model(cuda=False)
    refine_net = load_refinenet_model(cuda=False)
    all_boxes = []

    # Process each block in the given order
    for idx, (x, y, w, h) in enumerate(ordered_blocks):
        # Crop the block region from the image
        block_image = image[y:y+h, x:x+w]

        # Detect lines within the block
        prediction_result = get_prediction(
            image=block_image,
            craft_net=craft_net,
            refine_net=refine_net,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4,
            cuda=False,
            long_size=1280
        )

        # Adjust coordinates to the original image
        block_boxes = prediction_result["boxes"]
        adjusted_boxes = [box + np.array([x, y]) for box in block_boxes]

        # Sort lines within the block by their y-coordinate (top to bottom)
        adjusted_boxes.sort(key=lambda b: np.min(b[:, 1]))
        all_boxes.extend(adjusted_boxes)

    # Export all detected lines in reading order
    export_detected_regions(image=image, regions=all_boxes, rectify=True)
    export_extra_results(image=image, regions=all_boxes, output_path=output_dir)
    empty_cuda_cache()

    return all_boxes