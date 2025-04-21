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
import os
import torch
import numpy as np

def apply_craft_detection(image_path: str, output_dir: str) -> dict:
    """
    Applique CRAFT sur une image pour détecter les zones de texte.

    Args:
        image_path (str): Chemin de l'image.
        output_dir (str): Dossier de sortie pour les résultats.

    Returns:
        dict: Résultat brut de la détection contenant les boxes et heatmaps.
    """
    # Utiliser CUDA si disponible
    use_cuda = torch.cuda.is_available()
    print(f"Utilisation de {'GPU (CUDA)' if use_cuda else 'CPU'} pour CRAFT")

    # Lecture de l'image
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
        link_threshold=0.75,
        low_text=0.4,
        cuda=use_cuda,
        long_size=1280,
    )

    # # Suppression des lignes trop petites (en se basant sur l'aire et le ratio H/L)
    # areas = []
    # box_dims = []
    # for box in prediction["boxes"]:
    #     x_coords = box[:, 0]
    #     y_coords = box[:, 1]
    #     xmin, xmax = np.min(x_coords), np.max(x_coords)
    #     ymin, ymax = np.min(y_coords), np.max(y_coords)
    #     width = xmax - xmin
    #     height = ymax - ymin
    #     area = width * height
    #     areas.append(area)
    #     box_dims.append((box, width, height, area))

    # mean_area = np.mean(areas)
    # std_area = np.std(areas)
    # threshold = mean_area - 1.0 * std_area

    # filtered_boxes = []
    # for box, width, height, area in box_dims:
    #     ratio = height / width if width != 0 else 0
    #     # Conserver uniquement les lignes normales (ni trop petites ni trop fines)
    #     if area >= threshold and (ratio < 0.2 or ratio > 2.5):
    #         filtered_boxes.append(box)

    # prediction["boxes"] = filtered_boxes
    
    # Export des résultats
    os.makedirs(output_dir, exist_ok=True)

    # ⬇️ Nouvelle version : retourne les crops (ne les sauvegarde pas)
    crops = export_detected_regions(
        image=image,
        regions=prediction["boxes"],
        rectify=True
    )

    # ⬇️ Nouvelle version : exporte 4_detected_lines.png
    export_extra_results(
        image=image,
        regions=prediction["boxes"],
        output_path=output_dir
    )

    # Libération GPU
    empty_cuda_cache()

    print(f"Détection terminée. Résultats enregistrés dans {output_dir}")
    
    # Ajoute les crops dans le dictionnaire retourné si besoin
    prediction["crops"] = crops

    return prediction
