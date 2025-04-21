# -*- coding: utf-8 -*-
"""
Utilitaires pour regrouper les lignes en paragraphes et obtenir l'ordre
de lecture global (paragraphes + lignes) au moyen d'ORO (MLP).
"""
import numpy as np
import cv2
from collections import defaultdict
from pipeline.clustering import (
    convert_craft_boxes_to_rects,
    cluster_bounding_boxes_hdbscan,
    calculate_cluster_polygons,
)
from models.Order_Relation_Operator.reading_order import order_boxes
from pipeline.heatmap_ordering import get_ordered_polygons  



def poly_bbox(poly):
    xs, ys = zip(*poly)
    return [min(xs), min(ys), max(xs), max(ys)]


def group_lines(craft_boxes, use_cluster=True):
    """
    Regroupe les boîtes CRAFT en paragraphes (HDBSCAN) ou renvoie
    chaque ligne comme bloc individuel si use_cluster=False.
    Retourne :
        paragraph_polys : [poly,...] (liste de polygones)
        para_to_lines   : {pid:[idx_ligne,...]}
    """
    rects = convert_craft_boxes_to_rects(craft_boxes)

    if use_cluster and len(craft_boxes) > 1:
        centers, labels, _ = cluster_bounding_boxes_hdbscan(rects)
        cluster_polys = calculate_cluster_polygons(rects, labels, "convex")

        paragraph_polys, para_to_lines = [], defaultdict(list)
        for cid, poly in cluster_polys.items():
            for p in ([poly] if poly.geom_type == "Polygon" else poly.geoms):
                pid = len(paragraph_polys)
                paragraph_polys.append(
                    [(int(x), int(y)) for x, y in p.exterior.coords]
                )
                for idx, lbl in enumerate(labels):
                    if lbl == cid:
                        para_to_lines[pid].append(idx)
    else:
        paragraph_polys = [
            craft_boxes[i].astype(int).tolist() for i in range(len(craft_boxes))
        ]
        para_to_lines = {i: [i] for i in range(len(craft_boxes))}

    return paragraph_polys, para_to_lines


def reading_order(craft_boxes,
                  img_path,
                  oro_ckpt,
                  use_cluster=True,
                  decoder="count",
                  algo="oro"):
    """
    algo : 'oro' (MLP)  ou  'heatmap'
    Retourne (para_order, line_order, paragraph_polys)
    """
    paragraph_polys, para_to_lines = group_lines(craft_boxes, use_cluster)

    # -------- choix de l'algorithme d'ordre de paragraphes -------------
    if algo == "heatmap":
        ordered_polys, _ = get_ordered_polygons(
            paragraph_polys,  # polygones
            image_path=img_path,  
            sigma_x=200, sigma_y=200, tighten=0.9, reset_thresh=0.5
        )
        para_polys = ordered_polys
        para_order   = list(range(len(para_polys)))

    elif algo == "oro":
        para_rects = [poly_bbox(p) for p in paragraph_polys]
        if len(para_rects) > 1:
            para_order = order_boxes(
                para_rects, cv2.imread(img_path).shape, oro_ckpt, decoder
            )
        else:
            para_order = [0]
    else:
        raise ValueError(f"Unknown algo '{algo}' (use 'oro' or 'heatmap')")

    # ------------ ordre final des lignes ---------------------------------
    final_lines = []
    for pid in para_order:
        idxs = sorted(para_to_lines[pid],
                      key=lambda i: craft_boxes[i][:,1].mean())
        final_lines.extend(idxs)

    return para_order, final_lines, paragraph_polys