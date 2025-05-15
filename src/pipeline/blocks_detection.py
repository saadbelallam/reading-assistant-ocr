################
#    Step 2    #
################

#!/usr/bin/env python3
# 2_blocks_detection.py

"""
Reading‑order utilities 
---------------------------
• cluster_algo = 'hdbscan' (original) ou 'statistic'
• ordre de lecture via MLP ORO ou heatmap
"""

from collections import defaultdict


import cv2
import os
import numpy as np


# --- Clustering HDBSCAN -----------------------------------------
from utils.clustering import (
    convert_craft_boxes_to_rects,
    cluster_bounding_boxes_hdbscan,
    calculate_cluster_polygons,
    plot_cluster_polygons
)

# --- Clustering statistique --------------------------------
from utils.statistical_clustering import (
    hybrid_blocks_detection,      # -> (cluster_boxes, line2cid)
)



# ---------------------------------------------------------------------------
# 1. Regroupement des lignes en paragraphes avec statistical_learning
# ---------------------------------------------------------------------------


def detect_text_blocks_stat(image_path, result_path, craft_boxes, visualize=False):
    """
    Retourne
    --------
    paragraph_polys : liste de polygones de paragraphes
    para_to_lines   : dict {pid: [indices lignes]}
    """
    print("Running text blocks STAT ...")
    cluster_boxes, line2cid = hybrid_blocks_detection(craft_boxes, visualize, img_path=image_path,save_dir=result_path)
    # Polygone = rectangle du cluster.
    paragraph_polys = [
        [
            (int(x), int(y)),
            (int(x + w), int(y)),
            (int(x + w), int(y + h)),
            (int(x), int(y + h)),
        ]
        for x, y, w, h in cluster_boxes
    ]
    # Conversion mapping -> dict para_to_lines
    para_to_lines = defaultdict(list)
    for idx, cid in enumerate(line2cid):
        if cid >= 0:
            para_to_lines[cid].append(idx)
    return paragraph_polys, para_to_lines




def detect_text_blocks_hdbscan(image_path, save_dir, craft_boxes, visualize=False):
    """
    Détecte les blocs de texte à partir des boîtes CRAFT en utilisant HDBSCAN.

    Retourne
    --------
    paragraph_polys : liste de polygones de paragraphes
    para_to_lines   : dict {pid: [indices lignes]}
    """
    print("Running text blocks HDBSCAN ...")
    os.makedirs(save_dir, exist_ok=True)

    rects = convert_craft_boxes_to_rects(craft_boxes)
    _, labels, _ = cluster_bounding_boxes_hdbscan(rects)

    method = "bounding_box"  # ou "convex"
    cluster_polys = calculate_cluster_polygons(rects, labels, method=method)

    paragraph_polys, para_to_lines = [], defaultdict(list)
    for cid, poly in cluster_polys.items():
        pid = len(paragraph_polys)
        paragraph_polys.append([(int(x), int(y)) for x, y in poly])
        for idx, lbl in enumerate(labels):
            if lbl == cid:
                para_to_lines[pid].append(idx)

    if visualize:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        for polygon in paragraph_polys:
            pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
            color = tuple(int(c) for c in np.random.randint(0, 255, size=3))
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        filename = "3_detected_blocks.png"
        output_path = os.path.join(save_dir, filename)
        cv2.imwrite(output_path, img)
        print(f"Visualized blocks saved to {output_path}")

    return paragraph_polys, para_to_lines
