################
#    Step 3    #
################

import torch
import cv2
import os
import numpy as np
import math
import itertools
from typing import List, Tuple
from models.reading_order.model import ReadingOrderModel
from pipeline.blocks_ordering_XY import visualize_xycut_order

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
def load_reading_order_model(path="models/reading_order/best_model.pt"):
    model = ReadingOrderModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

model = load_reading_order_model()

# Predict full matrix of p(bi → bj)
def predict_parent_matrix(blocks):
    n = len(blocks)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            box1 = torch.tensor(blocks[i], dtype=torch.float32).to(device)
            box2 = torch.tensor(blocks[j], dtype=torch.float32).to(device)

            with torch.no_grad():
                prob = torch.sigmoid(model(box1.unsqueeze(0), box2.unsqueeze(0))).item()
                matrix[i][j] = prob
    return matrix

def held_karp_decoder(P):
    """
    P: numpy array of shape (n, n), where P[i][j] = probability that bi -> bj
    Returns: list of indices giving the best reading order
    """
    n = P.shape[0]
    log_cost = np.full((n, n), np.inf)

    eps = 1e-4  # to prevent log(0)
    for i in range(n):
        for j in range(n):
            if i != j:
                log_cost[i][j] = -math.log(max(P[i][j], eps))

    # DP table: dp[mask][i] = (cost, prev)
    dp = {}
    for i in range(n):
        dp[(1 << i, i)] = (0, -1)  # start at i with only i visited

    for subset_size in range(2, n + 1):
        for subset in itertools.combinations(range(n), subset_size):
            bits = sum([1 << x for x in subset])
            for j in subset:
                prev_bits = bits & ~(1 << j)
                best = (float('inf'), -1)
                for k in subset:
                    if k == j or (prev_bits, k) not in dp:
                        continue
                    cost = dp[(prev_bits, k)][0] + log_cost[k][j]
                    if cost < best[0]:
                        best = (cost, k)
                dp[(bits, j)] = best

    # Find best path ending at any node
    bits = (1 << n) - 1
    min_cost, end = min((dp[(bits, j)][0], j) for j in range(n))

    # Reconstruct path
    path = []
    while bits:
        path.append(end)
        bits, end = bits & ~(1 << end), dp[(bits, end)][1]
    path.reverse()
    return path

def decode_ranking(matrix):
    scores = matrix.sum(axis=1)  # score for each block
    order = np.argsort(-scores)  # higher score = earlier in reading order
    return order.tolist()


def reading_order_blocks(
    image_path: str,  
    blocks: List[List[Tuple[int, int]]],  
    output_dir: str = "tests/results"
) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """
    Orders text blocks using an iterative X-Y Cut algorithm.
    """
    print("Running blocks ordering ML ...")
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Convert contours to bounding boxes (x, y, w, h)
    boxes = [(int(block[0][0]), int(block[0][1]), int(block[2][0] - block[0][0]), int(block[2][1] - block[0][1])) for block in blocks]
    # Store image dimensions for normalization
    img_h, img_w = image.shape[:2]

    # Normalize boxes to 0–1000 range
    def normalize_box(box, width, height):
        x, y, w, h = box
        return [
            (x / width) * 1000,
            (y / height) * 1000,
            (w / width) * 1000,
            (h / height) * 1000
        ]

    normalized_boxes = [normalize_box(b, img_w, img_h) for b in boxes]

    # Order the blocks using X-Y cut
    P = predict_parent_matrix(normalized_boxes)
    
    # Decode using Held-Karp algorithm
    order = decode_ranking(P)
    # Reorder boxes based on the predicted order
    ordered_boxes = [boxes[i] for i in order]
    # Generate visualization
    vis_image = visualize_xycut_order(image.copy(), ordered_boxes)

    # Save results
    cv2.imwrite(f"{output_dir}/3_ordered_blocks.jpg", vis_image)

    return ordered_boxes, order