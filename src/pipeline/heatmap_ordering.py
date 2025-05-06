import cv2
import numpy as np
import math
import os
###############################################################################
#                           HELPER FUNCTIONS
###############################################################################

def read_polygons(filename):
    """
    Reads polygon coordinates from a text file.
    Each line should contain comma-separated integers.
    Example of one line:
       "115,20,660,26,660,54,115,48"
    Returns a list of polygons (each polygon is a list of (x, y) tuples).
    """
    polygons = []
    with open(filename, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = [p for p in line.split(",") if p.strip()]
            if len(parts) % 2 != 0:
                raise ValueError(f"Invalid coordinate count in line {line_number}: '{line}'")
            coords = list(map(int, parts))
            poly = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            polygons.append(poly)
    return polygons

def generate_cone_heatmap(image_shape, origin, sigma_x, sigma_y, angle):
    """
    Generates an anisotropic Gaussian heatmap from 'origin' with major/minor
    axes (sigma_x, sigma_y), rotated by 'angle'. Returns a 2D array [0..255].

    Additionally, limits the heatmap to the **lower half-circle**.
    """
    h, w = image_shape
    cx, cy = origin
    y_grid, x_grid = np.mgrid[0:h, 0:w]

    # Shift coords so (origin) becomes (0,0)
    x_shift = x_grid - cx
    y_shift = y_grid - cy

    # Rotate by -angle
    x_rot = x_shift * math.cos(angle) + y_shift * math.sin(angle)
    y_rot = -x_shift * math.sin(angle) + y_shift * math.cos(angle)

    # Compute heatmap values
    heatmap_float = np.exp(
        -0.5 * (
            (x_rot**2) / (sigma_x**2) +
            (y_rot**2) / (sigma_y**2)
        )
    )

    # **Limit to Lower Half Circle**
    heatmap_float[y_shift < 0] = 0  # Remove all pixels **above** origin

    # Normalize to [0, 255]
    max_val = np.max(heatmap_float)
    if max_val > 0:
        heatmap_float = (heatmap_float / max_val) * 255.0

    return heatmap_float.astype(np.uint8)


def polygon_mask(poly, image_shape):
    """
    Creates a binary mask (0/1) for a polygon on an image of shape=(height, width).
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask

def polygon_score(poly, heatmap):
    """
    Computes the average heat for the given polygon:
      sum of heatmap values inside the polygon / polygon area in pixels.
    """
    mask = polygon_mask(poly, heatmap.shape)
    heat = np.max(heatmap * mask)

    return heat

def min_distance_between_polygons(polyA, polyB):
    """
    Returns the minimal Euclidean distance between any point in polyA
    and any point in polyB.
    """
    min_dist = float("inf")
    for (xA, yA) in polyA:
        for (xB, yB) in polyB:
            dist = math.hypot(xB - xA, yB - yA)
            if dist < min_dist:
                min_dist = dist
    return min_dist

def bottom_left_point(poly):
    """
    Returns the left-most bottom point of a polygon in image coordinates:
      - "Bottom" means largest y
      - Among those, pick the smallest x (left-most).
    """
    max_y = max(pt[1] for pt in poly)
    min_x = min(pt[0] for pt in poly)
    return min_x , max_y

def top_left_point(poly):
    """
    Returns the point in the polygon closest to the lower end of the y = x line (first bisector).
    - "Lower end" of y=x => direction where x and y increase (bottom-right in image coordinates).
    - Prioritize points with the largest (x + y), then resolve ties by selecting the smallest x.
    """
    # Calculate x + y for all points to measure their position along the y = x line
    sums = [pt[0] + pt[1] for pt in poly]
    max_sum = max(sums)
    # Filter points with the largest x + y sum
    candidates = [pt for pt, s in zip(poly, sums) if s == max_sum]
    # Resolve ties by selecting the point with the smallest x (similar to original logic)
    return min(candidates, key=lambda pt: pt[0])

###############################################################################
#                   ORDERING FUNCTION USING POLYGON-TO-POLYGON DISTANCE
###############################################################################



def centroid(poly):
    """Calculate polygon centroid"""
    x = sum(p[0] for p in poly)/len(poly)
    y = sum(p[1] for p in poly)/len(poly)
    return (int(x), int(y))



def heatmap_ordering(polygons, image_shape,
                     init_origin=(0,0),
                     init_sigma_x=500,  # Wider horizontal spread
                     init_sigma_y=500,  # Narrower vertical spread
                     init_angle=0,
                     tighten_factor=0.85,
                     reset_threshold=0.3,
                     max_angle_deg=30):  # Maximum allowed angle

    remaining = polygons.copy()
    ordered = []
    heatmap_states = []
    prev_poly = None
    origin = init_origin
    sigma_x = init_sigma_x
    sigma_y = init_sigma_y
    angle = init_angle
    max_angle = math.radians(max_angle_deg)
    epsilon = 1e-6  # Small constant to avoid division by zero

    while remaining:
        # Generate heatmap
        heatmap = generate_cone_heatmap(image_shape, origin, sigma_x, sigma_y, angle)

        # Calculate scores
        scores = []
        for poly in remaining:
            heat = polygon_score(poly, heatmap)
            
            if prev_poly:
                # Compute distance between previous polygon and candidate polygon
                dist = min_distance_between_polygons(prev_poly, poly) + epsilon
                score = heat / dist  # Normalize by distance
            else:
                score = heat  # No distance for first polygon

            scores.append(score)

        # Find best candidate
        max_score = max(scores)
        if max_score < reset_threshold:
            # On définit l'intervalle horizontal [current_x, current_x + e]
            current_x = origin[0]
            e = 0.05 * image_shape[1]  # e initial : 5% de la largeur de l'image
            # On sélectionne les polygones dont le min des x (le point le plus à gauche) se trouve dans l'intervalle
            candidates = [p for p in remaining if current_x <= min(pt[0] for pt in p) <= current_x + e]
            
            # Debug : affichage des min_x de chaque polygone restant
            for p in remaining:
                min_x_val = min(pt[0] for pt in p)
            
            # Tant qu'aucun candidat n'est trouvé et que l'intervalle n'atteint pas le bord droit, on augmente e
            while not candidates and (current_x + e < image_shape[1]):
                e += 0.05 * image_shape[1]
                candidates = [p for p in remaining if current_x <= min(pt[0] for pt in p) <= current_x + e]
                for p in candidates:
                    min_x_val = min(pt[0] for pt in p)
            
            if candidates:
                # Parmi les candidats, on choisit celui dont le min des y (le plus haut dans l'image) est le plus petit
                next_poly = min(candidates, key=lambda p: min(pt[1] for pt in p))
            else:
                # Fallback : parmi tous les polygones restants, celui avec le plus petit min(y)
                next_poly = min(remaining, key=lambda p: min(pt[1] for pt in p))
        else:
            next_poly = remaining[np.argmax(scores)]

        



        # Update state
        ordered.append(next_poly)
        remaining.remove(next_poly)
        prev_poly = next_poly  # Track previous polygon for next iteration

        # Update heatmap parameters
        new_origin = min(next_poly, key=lambda pt: pt[0])  # Left-most point
        dx = new_origin[0] - origin[0]
        dy = new_origin[1] - origin[1]

        # Compute tightening factors dynamically
        shrink_x = 1 - abs(dx) / (abs(dx) + abs(dy) + epsilon)
        shrink_y = 1 - abs(dy) / (abs(dx) + abs(dy) + epsilon)

        sigma_x *= (tighten_factor + shrink_x * (1 - tighten_factor))
        sigma_y *= (tighten_factor + shrink_y * (1 - tighten_factor))

        # Limit angle to horizontal emphasis
        new_angle = math.atan2(dy, dx)
        if abs(new_angle) > max_angle:
            new_angle = math.copysign(max_angle, new_angle)

        origin = new_origin
        angle = new_angle

        # Store heatmap state for visualization
        heatmap_states.append((origin, sigma_x, sigma_y, angle))

    return ordered, heatmap_states


###############################################################################
#                           IMAGE ANNOTATION FUNCTIONS
###############################################################################

def draw_polygon(img, poly, color=(0,255,0), thickness=2, idx=None):
    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, color, thickness)
    if idx is not None:
        centroid = np.mean(pts[:,0,:], axis=0).astype(int)
        cv2.putText(img, str(idx), tuple(centroid),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

def annotate_image(image_path, ordered_polygons, output_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image: " + image_path)
    for idx, poly in enumerate(ordered_polygons):
        draw_polygon(img, poly, color=(0,255,0), thickness=2, idx=idx+1)
    cv2.imwrite(output_path, img)

def annotate_image_with_heatmaps(image_path, ordered_polygons, heatmap_states,
                                 init_sigma_x, init_sigma_y, output_folder):
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


def get_ordered_polygons(image_path, polygons,result_path, 
                         annotate=True,
                         ):
    """
    Applique l'algorithme heatmap_ordering aux boxes détectées par CRAFT.

    Paramètres :
        boxes : Liste de boîtes détectées (array ou liste de listes de points (x, y)).
        image_path : Chemin vers l'image utilisée.
        annotate : Booléen, si True génère les images annotées et les heatmaps.
        final_output : Nom du fichier de sortie pour l’image annotée.
        heatmaps_folder : Dossier où stocker les heatmaps si annotate est True.

    Retourne :
        ordered_polygons : liste ordonnée des polygones.
    """
    # Conversion de polygons en liste de tuples (x, y)
    polygons = [list(map(tuple, poly)) for poly in polygons]
    # Chargement de l'image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    image_shape = img.shape[:2]  # (height, width)

    # Origine initiale = top-left polygon
    first_poly = min(
        polygons,
        key=lambda p: (min(pt[1] for pt in p), min(pt[0] for pt in p))
    )
    init_origin = top_left_point(first_poly)

    # Exécution de l’algorithme d’ordre
    ordered_polygons, heatmap_states = heatmap_ordering(
        polygons, image_shape,
        init_origin=init_origin,
        init_sigma_x=200,
        init_sigma_y=200,
        init_angle=0,
        tighten_factor=0.9,
        reset_threshold=0.5
    )

    # Annotations optionnelles
    if annotate:
        result_path, heatmaps_folder = result_path + "/final_annotated_output.png", result_path + "/heatmap_overlays"
        annotate_image(image_path, ordered_polygons, result_path)
        annotate_image_with_heatmaps(
            image_path,
            ordered_polygons,
            heatmap_states,
            200,
            200,
            heatmaps_folder
        )

    order_idx = [ordered_polygons.index(p) for p in polygons]

    return None, order_idx

