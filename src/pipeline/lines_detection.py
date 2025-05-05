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
import torch

def detect_lines(image_path, output_dir):
    print("Running lines detection ...")
    image = read_image(image_path)
    prediction_result = apply_craft_detection(image_path)
    all_boxes = prediction_result["boxes"]

    # Export all lines in reading order
    export_detected_regions(image=image, regions=all_boxes, rectify=True)
    if output_dir:
        export_extra_results(image=image, regions=all_boxes, output_path=output_dir)
    empty_cuda_cache()

    return all_boxes

def apply_craft_detection(image_path):
    """
    Applique CRAFT sur une image pour détecter les zones de texte.

    Args:
        image_path (str): Chemin de l'image.

    Returns:
        dict: Résultat brut de la détection contenant les boxes et heatmaps.
    """
    # Utiliser CUDA si disponible
    use_cuda = torch.cuda.is_available()
    image = read_image(image_path)
    
    # Chargement des modèles
    craft_net = load_craftnet_model(cuda=use_cuda)
    refine_net = load_refinenet_model(cuda=use_cuda)

    # Prédiction
    prediction = get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.7,
        link_threshold=0.7,
        low_text=0.4,
        cuda=use_cuda,
        long_size=1280,
    )

    return prediction