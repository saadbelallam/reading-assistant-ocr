import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import numpy as np
from lines_detection import apply_craft_detection

def quads_to_rectangles(quadrilaterals):
    """Convert CRAFT quadrilaterals to [x, y, w, h] rectangles"""
    rectangles = []
    for quad in quadrilaterals:
        x_coords = quad[:, 0]
        y_coords = quad[:, 1]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        w = x_max - x_min
        h = y_max - y_min
        rectangles.append([x_min, y_min, w, h])
    return rectangles

def calculate_dynamic_thresholds(boxes):
    """Calculate thresholds based on box statistics"""
    # Extract all heights and widths
    heights = [h for _, _, _, h in boxes]
    widths = [w for _, _, w, _ in boxes]
    
    # Calculate median dimensions
    median_h = np.median(heights)
    median_w = np.median(widths)
    
    # Calculate interquartile ranges for robustness
    q1_h, q3_h = np.percentile(heights, [25, 75])
    iqr_h = q3_h - q1_h
    
    q1_w, q3_w = np.percentile(widths, [25, 75])
    iqr_w = q3_w - q1_w
    
    # Dynamic multipliers based on distribution characteristics
    alpha = 1.75
    v_multiplier = alpha + (iqr_h / median_h)
    h_multiplier = (2-alpha) * (iqr_w / median_w)

    return v_multiplier, h_multiplier

def close_enough(b1, b2, v_multiplier=1.5, h_multiplier=0.25):
    """
    b1, b2: (x, y, w, h)
    Return True if b2 lies “close” to b1, using simple relative gaps.
    Symmetric in b1, b2.
    """

    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    # compute center distances
    dy = abs((y1 + h1/2) - (y2 + h2/2))
    dx = abs((x1 + w1/2) - (x2 + w2/2))

    # compute effective thresholds
    v_thresh = v_multiplier * min(h1, h2)
    h_thresh = h_multiplier * min(w1, w2)

    return (dy <= v_thresh) and (dx <= h_thresh)

def cluster_text_regions(boxes):
    """Recursive implementation of text region clustering"""
    N = len(boxes)
    clustered = [False] * N
    clusters = []
    v_multiplier, h_multiplier = calculate_dynamic_thresholds(boxes)
    #print(f"v_multiplier: {v_multiplier}, h_multiplier: {h_multiplier}")

    def explore_cluster(current_idx, cluster, clustered):
        """Recursively explore neighboring boxes"""
        # Add current box to cluster
        cluster.append(current_idx)
        clustered[current_idx] = True

        # Find all unclustered neighbors
        neighbors = []
        for j in range(N):
            if not clustered[j] and close_enough(boxes[current_idx], boxes[j], v_multiplier, h_multiplier):
                neighbors.append(j)

        # Recursively explore each neighbor
        for neighbor in neighbors:
            if not clustered[neighbor]:
                explore_cluster(neighbor, cluster, clustered)

        return cluster

    for i in range(N):
        if not clustered[i]:
            current_cluster = []
            explore_cluster(i, current_cluster, clustered)
            clusters.append(current_cluster)

    return clusters

def show_clusters(image_path, boxes, clusters):
    
    if not boxes or not clusters:
        print("No boxes or clusters to display!")
        return
        
    # Load and display the image
    try:
        img = mpimg.imread(image_path)
        height, width = img.shape[:2]
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    colors = [(random.random(), random.random(), random.random()) for _ in clusters]
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Display the image
    ax.imshow(img)
    
    # Set axis limits based on image dimensions
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    
    # Draw the rectangles
    for color, cluster in zip(colors, clusters):
        for i in cluster:
            x, y, w, h = boxes[i]
            rect = plt.Rectangle((x, y), w, h, 
                                edgecolor=color, 
                                facecolor=color + (0.3,),
                                linewidth=2,
                                fill=False)
            ax.add_patch(rect)
    
    plt.title(f"{len(clusters)} clusters of {len(boxes)} boxes")
    plt.show()

def generate_cluster_bounding_boxes(boxes, clusters):
    """
    Génère un rectangle englobant pour chaque cluster.
    """
    cluster_boxes = []

    for cluster in clusters:
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = float('-inf'), float('-inf')

        for idx in cluster:
            box = boxes[idx]
            # Si box est un tableau NumPy, on force les éléments en float
            x = float(box[0])
            y = float(box[1])
            w = float(box[2])
            h = float(box[3])

            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        cluster_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])

    return cluster_boxes

def calculate_overlap_area(box1, box2):
    """Calculate the area of overlap between two boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate coordinates
    box1_right = x1 + w1
    box1_bottom = y1 + h1
    box2_right = x2 + w2
    box2_bottom = y2 + h2
    
    # Calculate overlap coordinates
    overlap_x1 = max(x1, x2)
    overlap_y1 = max(y1, y2)
    overlap_x2 = min(box1_right, box2_right)
    overlap_y2 = min(box1_bottom, box2_bottom)
    
    # Calculate overlap area
    overlap_width = max(0, overlap_x2 - overlap_x1)
    overlap_height = max(0, overlap_y2 - overlap_y1)
    
    return overlap_width * overlap_height

def is_partially_contained(box1, box2, overlap_threshold=0.75):
    """
    Check if box1 is partially contained in box2 based on overlap area.
    overlap_threshold: minimum ratio of box1's area that must overlap with box2
    """
    # Calculate areas
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    
    # Calculate overlap area
    overlap_area = calculate_overlap_area(box1, box2)
    
    # Check if overlap exceeds threshold relative to either box
    return (overlap_area >= overlap_threshold * area1) or (overlap_area >= overlap_threshold * area2)

def merge_contained_clusters(cluster_boxes, overlap_threshold=0.75):
    """Merge clusters that partially overlap based on threshold"""
    n = len(cluster_boxes)
    merged_indices = set()
    final_boxes = []
    
    for i in range(n):
        if i in merged_indices:
            continue
            
        current_box = cluster_boxes[i]
        to_merge = [current_box]
        
        # Find all boxes that overlap with current box
        for j in range(i+1, n):
            if j in merged_indices:
                continue
                
            other_box = cluster_boxes[j]
            if is_partially_contained(current_box, other_box, overlap_threshold):
                to_merge.append(other_box)
                merged_indices.add(j)
        
        # Merge all overlapping boxes
        if len(to_merge) > 1:
            # Calculate the union of all boxes to merge
            x_min = min(box[0] for box in to_merge)
            y_min = min(box[1] for box in to_merge)
            x_max = max(box[0] + box[2] for box in to_merge)
            y_max = max(box[1] + box[3] for box in to_merge)
            
            merged_box = [x_min, y_min, x_max - x_min, y_max - y_min]
            final_boxes.append(merged_box)
        else:
            final_boxes.append(current_box)
    
    return final_boxes

def blocks_detection(boxes, image_path=None, visualize=False):
    """
    Main function to cluster text regions and visualize them.
    """
    boxes = quads_to_rectangles(boxes)
    clusters = cluster_text_regions(boxes)
    cluster_boxes = generate_cluster_bounding_boxes(boxes, clusters)
    cluster_boxes = merge_contained_clusters(cluster_boxes)
    if visualize:
        # Visualize the bounding boxes of clusters (optional)
        try:
            img = mpimg.imread(image_path)
            height, width = img.shape[:2]
        except Exception as e:
            print(f"Erreur lors du chargement de l'image : {e}")
            exit()

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

        for rect in cluster_boxes:
            x, y, w, h = rect
            outline = plt.Rectangle((x, y), w, h, 
                                    edgecolor='green', 
                                    linewidth=1.0, 
                                    linestyle='-', 
                                    fill=False)
            ax.add_patch(outline)
        plt.title("Rectangles englobants des clusters")
        plt.show()

    return cluster_boxes

# Example usage
#if __name__ == "__main__":
#    path = "/home/ensimag/projet_specialite/tests/images/Documents_imprimes/easy/mcolumn.jpg"
#    boxes = apply_craft_detection(path)["boxes"]
#    clusters = blocks_detection(boxes, path, True)
    