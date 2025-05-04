"""
Reading‑order utilities 
---------------------------
• cluster_algo = 'hdbscan' (original) ou 'statistic'
• ordre de lecture via MLP ORO ou heatmap
"""

from collections import defaultdict
from typing import List, Tuple

import cv2
import numpy as np

# --- Clustering HDBSCAN -----------------------------------------
from pipeline.clustering import (
    convert_craft_boxes_to_rects,
    cluster_bounding_boxes_hdbscan,
    calculate_cluster_polygons,
)

# --- Clustering statistique --------------------------------
from pipeline.statistical_clustering import (
    hybrid_blocks_detection,      # -> (cluster_boxes, line2cid)
)

# --- Modules pour l'ordre de lecture --------------------------------------
from models.Order_Relation_Operator.reading_order import order_boxes   # MLP ORO
from pipeline.heatmap_ordering import get_ordered_polygons             # Heatmap

# ---------------------------------------------------------------------------
# Helpers 
# ---------------------------------------------------------------------------

def poly_bbox(poly: List[Tuple[int, int]]) -> List[int]:
    """Bounding‑box d'un polygone [(x,y), ...] -> [x_min, y_min, x_max, y_max]."""
    xs, ys = zip(*poly)
    return [min(xs), min(ys), max(xs), max(ys)]



# ---------------------------------------------------------------------------
# 1. Regroupement des lignes en paragraphes
# ---------------------------------------------------------------------------

def group_lines(
    craft_boxes: np.ndarray,
    use_cluster: bool = True,
    cluster_algo: str = "hdbscan"):
    """
    Retourne
    --------
    paragraph_polys : liste de polygones de paragraphes
    para_to_lines   : dict {pid: [indices lignes]}
    """
    rects = convert_craft_boxes_to_rects(craft_boxes)

    # --- Cas sans clustering (ou une seule ligne) -------------------------
    if not use_cluster or len(craft_boxes) <= 1:
        polys = [craft_boxes[i].astype(int).tolist() for i in range(len(craft_boxes))]
        return polys, {i: [i] for i in range(len(craft_boxes))}

    # --- 1) Algorithme HDBSCAN ---------------------------------
    if cluster_algo == "hdbscan":
        _, labels, _ = cluster_bounding_boxes_hdbscan(rects)
        cluster_polys = calculate_cluster_polygons(rects, labels, "convex")

        paragraph_polys, para_to_lines = [], defaultdict(list)
        for cid, poly in cluster_polys.items():
            for p in [poly] :
                pid = len(paragraph_polys)
                paragraph_polys.append(
                    [(int(x), int(y)) for x, y in p.exterior.coords]
                )
                for idx, lbl in enumerate(labels):
                    if lbl == cid:
                        para_to_lines[pid].append(idx)
        return paragraph_polys, para_to_lines

    # --- 2) Algorithme statistique two‑stage ------------------------------
    elif cluster_algo == "statistic":
        cluster_boxes, line2cid = hybrid_blocks_detection(
            craft_boxes, visualize=False
        )

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

    else:
        raise ValueError("cluster_algo must be 'hdbscan' or 'statistic'")

# ---------------------------------------------------------------------------
# 2. Ordre de lecture global
# ---------------------------------------------------------------------------

def reading_order(
    craft_boxes: np.ndarray,
    img_path: str,
    oro_ckpt: str,
    *,
    use_cluster: bool = True,
    decoder: str = "count",
    algo: str = "oro",
    cluster_algo: str = "hdbscan",
) -> Tuple[List[int], List[int], List[List[Tuple[int, int]]]]:
    """
    Parameters
    ----------
    img_path      : image originale (pour ORO/heatmap).
    oro_ckpt      : checkpoint du MLP Order‑Relation Operator.
    algo          : 'oro' (MLP) ou 'heatmap' pour l’ordre des paragraphes.
    cluster_algo  : 'hdbscan' ou 'statistic' pour regrouper les lignes.

    Returns
    -------
    para_order  : ordre des paragraphes (indices dans paragraph_polys)
    line_order  : ordre global des lignes (indices dans craft_boxes)
    paragraph_polys : polygones des paragraphes
    """
    # 1) Clustering des lignes
    paragraph_polys, para_to_lines = group_lines(
        craft_boxes, use_cluster, cluster_algo
    )

    # 2) Ordre des paragraphes
    if algo == "heatmap":
        ordered_polys, _ = get_ordered_polygons(
            paragraph_polys,
            image_path=img_path,
            sigma_x=200,
            sigma_y=200,
            tighten=0.9,
            reset_thresh=0.5,
        )
        para_order = list(range(len(ordered_polys)))
        paragraph_polys = ordered_polys  # on garde l’ordre renvoyé
    elif algo == "oro":
        para_rects = [poly_bbox(p) for p in paragraph_polys]
        if len(para_rects) > 1:
            img_shape = cv2.imread(img_path).shape
            para_order = order_boxes(para_rects, img_shape, oro_ckpt, decoder)
        else:
            para_order = [0]
    else:
        raise ValueError("algo must be 'oro' or 'heatmap'")

    # 3) Ordre des lignes à l’intérieur de chaque paragraphe
    line_order: List[int] = []
    for pid in para_order:
        # Tri vertical par centre Y
        idxs = sorted(para_to_lines.get(pid, []),
                      key=lambda i: craft_boxes[i][:, 1].mean())
        line_order.extend(idxs)

    return para_order, line_order, paragraph_polys
