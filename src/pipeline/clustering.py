import numpy as np
import hdbscan
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
from collections import defaultdict


def craft_box_to_rect(box):
    x_coords = box[:, 0]
    y_coords = box[:, 1]
    return int(np.min(x_coords)), int(np.min(y_coords)), int(np.max(x_coords)), int(np.max(y_coords))


def cluster_bounding_boxes_hdbscan(
    bounding_boxes,
    min_cluster_size=2,
    min_samples=1,
    proba_threshold=0.6
):
    centers = np.array([[xmin, ymin] for (xmin, ymin, xmax, ymax) in bounding_boxes])

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='cityblock'
    )
    labels = clusterer.fit_predict(centers)
    probabilities = clusterer.probabilities_

    max_label = labels.max()
    for i in range(len(labels)):
        if labels[i] == -1 or probabilities[i] < proba_threshold:
            max_label += 1
            labels[i] = max_label

    return centers, labels, probabilities


def calculate_cluster_polygons(bounding_boxes, labels, method="convex"):
    cluster_polygons = {}
    clusters_dict = defaultdict(list)

    for bbox, label in zip(bounding_boxes, labels):
        clusters_dict[label].append(bbox)

    for cluster_id, bb_list in clusters_dict.items():
        poly_list = [box(xmin, ymin, xmax, ymax) for (xmin, ymin, xmax, ymax) in bb_list]
        merged = unary_union(poly_list)
        if method == "convex":
            merged = merged.convex_hull
        cluster_polygons[cluster_id] = merged

    return cluster_polygons


def plot_clusters_on_image(bounding_boxes, labels, image_path):
    image = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)

    unique_labels = set(labels)
    color_map = {label: (random.random(), random.random(), random.random()) for label in unique_labels}

    for (xmin, ymin, xmax, ymax), cluster_id in zip(bounding_boxes, labels):
        color = color_map[cluster_id]
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)

    ax.set_title("Clustering des bounding boxes (HDBSCAN)")
    plt.show()


def plot_points_clusters(centers, labels):
    unique_labels = set(labels)
    color_map = {label: (random.random(), random.random(), random.random()) for label in unique_labels}

    plt.figure(figsize=(8, 6))
    for point, cluster_id in zip(centers, labels):
        plt.scatter(point[0], point[1], color=color_map[cluster_id], label=f'Cluster {cluster_id}' if f'Cluster {cluster_id}' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.xlabel('X (xmin)')
    plt.ylabel('Y (ymin)')
    plt.title('Visualisation des points (xmin,ymin) par cluster (HDBSCAN)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_cluster_polygons(cluster_polygons, image_path):
    image = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)

    for cluster_id, poly in cluster_polygons.items():
        color = (random.random(), random.random(), random.random())
        if poly.geom_type == 'Polygon':
            polys = [poly]
        else:
            polys = poly.geoms

        for p in polys:
            x, y = p.exterior.xy
            plt.fill(x, y, alpha=0.4, color=color)
            plt.plot(x, y, color=color, linewidth=2)

    ax.set_title("Polygones de clusters via Shapely (HDBSCAN)")
    plt.savefig("outputs/plot_clusters.png")


def convert_craft_boxes_to_rects(craft_boxes):
    return [craft_box_to_rect(box) for box in craft_boxes]
