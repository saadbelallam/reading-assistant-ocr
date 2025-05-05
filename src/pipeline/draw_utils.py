# -*- coding: utf-8 -*-
import cv2, os, numpy as np


def draw_polygons(img_path, polygons, out_path, color=(0, 255, 0)):
    img = cv2.imread(img_path)
    for i, poly in enumerate(polygons, 1):
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, color, 2)
        cx, cy = pts[:, 0, 0].mean(), pts[:, 0, 1].mean()
        cv2.putText(img, str(i), (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)
    return out_path


def draw_lines(img_path, craft_boxes, line_order, out_path):
    img = cv2.imread(img_path)
    for rank, idx in enumerate(line_order, 1):
        poly = craft_boxes[idx].astype(int).reshape((-1, 1, 2))
        cv2.polylines(img, [poly], True, (255, 0, 0), 2)
        cx, cy = poly[:, 0, 0].mean(), poly[:, 0, 1].mean()
        cv2.putText(img, str(rank), (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)
    return out_path
