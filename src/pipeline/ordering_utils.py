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
    algo: str = "oro",          # ordre des paragraphes : 'oro' ou 'heatmap'
    line_algo: str = "heatmap",# ordre des lignes      : 'vertical' ou 'heatmap'
    cluster_algo: str = "statistic", # 'hdbscan' ou 'statistic'
):
    # 1) Regrouper les lignes en paragraphes --------------------------------
    paragraph_polys, para_to_lines = group_lines(
        craft_boxes, use_cluster, cluster_algo
    )

    # Outil pour transformer rapidement un poly (liste de tuples) en hashable
    to_key = lambda poly: tuple(np.ravel(poly))

    # 2) Ordre des paragraphes ----------------------------------------------
    if algo == "heatmap":
        ordered_polys = get_ordered_polygons(
            paragraph_polys,
            image_path=img_path,
            annotate=False,
            final_output=None,
            heatmaps_folder=None,
        )
        # convertir polygone → pid
        poly2pid = {to_key(p): i for i, p in enumerate(paragraph_polys)}
        para_order = [poly2pid[to_key(p)] for p in ordered_polys]
    elif algo == "oro":
        para_rects = [poly_bbox(p) for p in paragraph_polys]
        if len(para_rects) > 1:
            H, W = cv2.imread(img_path).shape[:2]
            para_order = order_boxes(para_rects, (H, W, 3), oro_ckpt, decoder)
        else:
            para_order = [0]
    else:
        raise ValueError("algo must be 'oro' or 'heatmap'")

    # 3) Ordre des lignes dans chaque paragraphe ----------------------------
    line_order = []
    for pid in para_order:
        idxs = para_to_lines.get(pid, [])
        if len(idxs) <= 1:
            line_order.extend(idxs)
            continue

        if line_algo == "vertical":
            ordered = sorted(idxs, key=lambda i: craft_boxes[i][:, 1].mean())

        elif line_algo == "heatmap":
            # polygones des lignes de ce paragraphe
            line_polys = [craft_boxes[i].astype(int).tolist() for i in idxs]
            ordered_polys = get_ordered_polygons(
                line_polys,
                image_path=img_path,
                annotate=False,
                final_output=None,
                heatmaps_folder=None,
            )
            poly2idx = {to_key(p): i for p, i in zip(line_polys, idxs)}
            ordered = [poly2idx[to_key(p)] for p in ordered_polys]

        else:
            raise ValueError("line_algo must be 'vertical' or 'heatmap'")

        line_order.extend(ordered)

    return para_order, line_order, paragraph_polys

