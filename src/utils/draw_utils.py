import cv2
import numpy as np
import os
from typing import List

def draw_lines(image_path: str,
               ordered_lines: List[np.ndarray],
               order: List[int],
               output_dir: str,
               thickness: int = 2,
               show_index: bool = True) -> None:
    """
    Dessine les lignes (bounding boxes) détectées sur l'image, dans l'ordre spécifié.

    Paramètres :
    - image_path : chemin de l'image d'entrée
    - ordered_lines : liste de boîtes (chaque boîte est un np.ndarray de forme (4,2))
    - order : liste des indices des blocs dans l'ordre de lecture
    - output_dir : dossier où sauvegarder l'image avec les tracés
    - thickness : épaisseur des traits
    - show_index : afficher ou non l'indice de chaque ligne
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Couleurs fixes : bleu pour les lignes, rouge pour les indices
    line_color = (255, 0, 0)    # Bleu (BGR)
    text_color = (0, 0, 255)    # Rouge (BGR)

    for idx, box in enumerate(ordered_lines):
        pts = np.round(box).astype(np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=line_color, thickness=thickness)

        if show_index:
            # Afficher l’indice de la ligne (dans l’ordre)
            text_pos = tuple(pts[np.argmin(pts[:, 1])])  # coin supérieur
            cv2.putText(image, str(idx), (text_pos[0], text_pos[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

    output_path = os.path.join(output_dir, "4_ordered_lines_overlay.jpg")
    cv2.imwrite(output_path, image)
    print(f"[draw_lines] Image enregistrée : {output_path}")
