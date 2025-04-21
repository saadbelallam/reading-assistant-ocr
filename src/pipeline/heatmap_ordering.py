# pipeline/heatmap_ordering.py
"""
Algorithme heuristique 'heat‑map cone' pour ordonner des polygones.
On exporte : get_ordered_polygons(...)
"""
import cv2
import numpy as np
import math
import os

###############################################################################
#                           HELPER FUNCTIONS
###############################################################################

def generate_cone_heatmap(image_shape, origin, sigma_x, sigma_y, angle):
    h, w = image_shape
    cx, cy = origin
    y_grid, x_grid = np.mgrid[0:h, 0:w]

    x_shift = x_grid - cx
    y_shift = y_grid - cy

    x_rot = x_shift * math.cos(angle) + y_shift * math.sin(angle)
    y_rot = -x_shift * math.sin(angle) + y_shift * math.cos(angle)

    heatmap_float = np.exp(
        -0.5 * (
            (x_rot*2) / (sigma_x*2) +
            (y_rot*2) / (sigma_y*2)
        )
    )
    heatmap_float[y_shift < 0] = 0

    max_val = np.max(heatmap_float)
    if max_val > 0:
        heatmap_float = (heatmap_float / max_val) * 255.0

    return heatmap_float.astype(np.uint8)

def polygon_mask(poly, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask

def polygon_score(poly, heatmap):
    mask = polygon_mask(poly, heatmap.shape)
    heat = np.max(heatmap * mask)
    return heat

def min_distance_between_polygons(polyA, polyB):
    min_dist = float("inf")
    for (xA, yA) in polyA:
        for (xB, yB) in polyB:
            dist = math.hypot(xB - xA, yB - yA)
            if dist < min_dist:
                min_dist = dist
    return min_dist

def top_left_point(poly):
    # Retourne le point réellement en haut à gauche (min x, puis min y)
    return min(poly, key=lambda pt: (pt[0], pt[1]))


def draw_polygon(img, poly, color=(0,255,0), thickness=2, idx=None):
    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, color, thickness)
    if idx is not None:
        cent = np.mean(pts[:, 0, :], axis=0).astype(int)
        cv2.putText(img, str(idx), tuple(cent),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

def annotate_image(image_path, ordered_polygons, output_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image: " + image_path)
    for idx, poly in enumerate(ordered_polygons):
        draw_polygon(img, poly, color=(0,255,0), thickness=2, idx=idx+1)
    cv2.imwrite(output_path, img)
    print(f"Final annotated image saved to: {output_path}")

def annotate_image_with_heatmaps(image_path, ordered_polygons, heatmap_states, init_sigma_x, init_sigma_y, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    base_img = cv2.imread(image_path)
    if base_img is None:
        raise ValueError("Could not load image: " + image_path)
    for i, (origin, s_x, s_y, angle) in enumerate(heatmap_states, start=1):
        img_copy = base_img.copy()
        heat = generate_cone_heatmap(img_copy.shape[:2], origin, s_x, s_y, angle)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(heat_color, 0.3, img_copy, 0.7, 0)
        for idx, poly in enumerate(ordered_polygons):
            draw_polygon(overlay, poly, color=(0,255,0), thickness=2, idx=idx+1)
        out_path = os.path.join(output_folder, f"heatmap_state_{i}.jpg")
        cv2.imwrite(out_path, overlay)
        print(f"Saved overlay for iteration {i}: {out_path}")

def heatmap_ordering(polygons, image_shape,
                     init_origin=(0,0),
                     init_sigma_x=500,
                     init_sigma_y=500,
                     init_angle=0,
                     tighten_factor=0.85,
                     reset_threshold=0.3,
                     max_angle_deg=30):

    remaining = polygons.copy()
    ordered = []
    heatmap_states = []
    prev_poly = None
    origin = init_origin
    sigma_x = init_sigma_x
    sigma_y = init_sigma_y
    angle = init_angle
    max_angle = math.radians(max_angle_deg)
    epsilon = 1e-6

    while remaining:
        heatmap = generate_cone_heatmap(image_shape, origin, sigma_x, sigma_y, angle)

        scores = []
        for poly in remaining:
            heat = polygon_score(poly, heatmap)
            if prev_poly:
                dist = min_distance_between_polygons(prev_poly, poly) + epsilon
                score = heat / dist
            else:
                score = heat
            scores.append(score)

        max_score = max(scores)
        if max_score < reset_threshold:
            current_x = origin[0]
            e = 0.05 * image_shape[1]
            candidates = [p for p in remaining if current_x <= min(pt[0] for pt in p) <= current_x + e]
            while not candidates and (current_x + e < image_shape[1]):
                e += 0.05 * image_shape[1]
                candidates = [p for p in remaining if current_x <= min(pt[0] for pt in p) <= current_x + e]
            if candidates:
                next_poly = min(candidates, key=lambda p: min(pt[1] for pt in p))
            else:
                next_poly = min(remaining, key=lambda p: min(pt[1] for pt in p))
            sigma_x = init_sigma_x
            sigma_y = init_sigma_y
            angle = init_angle
            origin = init_origin
        else:
            next_poly = remaining[np.argmax(scores)]

        ordered.append(next_poly)
        remaining.remove(next_poly)
        prev_poly = next_poly

        new_origin = min(next_poly, key=lambda pt: pt[0])
        dx = new_origin[0] - origin[0]
        dy = new_origin[1] - origin[1]
        shrink_x = 1 - abs(dx) / (abs(dx) + abs(dy) + epsilon)
        shrink_y = 1 - abs(dy) / (abs(dx) + abs(dy) + epsilon)
        sigma_x *= (tighten_factor + shrink_x * (1 - tighten_factor))
        sigma_y *= (tighten_factor + shrink_y * (1 - tighten_factor))
        new_angle = math.atan2(dy, dx)
        if abs(new_angle) > max_angle:
            new_angle = math.copysign(max_angle, new_angle)
        origin = new_origin
        angle = new_angle
        heatmap_states.append((origin, sigma_x, sigma_y, angle))

    return ordered, heatmap_states

def get_ordered_polygons(polygons, image_path,
                         sigma_x=200, sigma_y=200,
                         angle=0, tighten=0.9, reset_thresh=0.5):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    image_shape = img.shape[:2]
    first_poly = min(polygons, key=lambda p: (min(pt[1] for pt in p), min(pt[0] for pt in p)))
    init_origin = top_left_point(first_poly)

    ordered_polygons, heatmap_states = heatmap_ordering(
        polygons, image_shape,
        init_origin=init_origin,
        init_sigma_x=sigma_x,
        init_sigma_y=sigma_y,
        init_angle=angle,
        tighten_factor=tighten,
        reset_threshold=reset_thresh
    )
    return ordered_polygons, heatmap_states