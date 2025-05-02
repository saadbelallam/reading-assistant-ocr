import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src.pipeline.lines_detection_v2 import apply_craft_detection
import os

# ---------------------------------------------------------------------------
# 1. Utils communs 
# ---------------------------------------------------------------------------

def quads_to_rectangles(quads):
    rectangles = []
    for quad in quads:
        x_min, y_min = np.min(quad[:, 0]), np.min(quad[:, 1])
        w = np.max(quad[:, 0]) - x_min
        h = np.max(quad[:, 1]) - y_min
        rectangles.append([x_min, y_min, w, h])
    return rectangles


def boxes_from_clusters(boxes, clusters):
    res = []
    for cl in clusters:
        xs = [boxes[i][0] for i in cl]
        ys = [boxes[i][1] for i in cl]
        xe = [boxes[i][0] + boxes[i][2] for i in cl]
        ye = [boxes[i][1] + boxes[i][3] for i in cl]
        res.append([min(xs), min(ys), max(xe) - min(xs), max(ye) - min(ys)])
    return res

# ---------------------------------------------------------------------------
# 2. Overlap & containment 
# ---------------------------------------------------------------------------

def overlap(a, b):
    x1, y1, w1, h1 = a
    x2, y2, w2, h2 = b
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    return iw * ih


def contains(box1, box2, overlap_threshold=0.75):
    """
    Check if box1 is partially contained in box2 based on overlap area.
    overlap_threshold: minimum ratio of box1's area that must overlap with box2
    """
    # Calculate areas
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    
    # Calculate overlap area
    overlap_area = overlap(box1, box2)
    
    # Check if overlap exceeds threshold relative to either box
    return (overlap_area >= overlap_threshold * area1) or (overlap_area >= overlap_threshold * area2)


# ---------------------------------------------------------------------------
# 3. Stage 1 – edge‑distance 
# ---------------------------------------------------------------------------

def median_height(boxes):
    return np.median([h for *_, h in boxes])


def ned(b1, b2, mh):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    dx = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
    dy = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
    return np.hypot(dx, dy) / mh


def cluster_edge(boxes, thr):
    mh = median_height(boxes)
    N = len(boxes)
    seen = [False] * N
    clusters = []
    def dfs(i, cur):
        seen[i] = True
        cur.append(i)
        for j in range(N):
            if not seen[j] and ned(boxes[i], boxes[j], mh) <= thr:
                dfs(j, cur)
    for i in range(N):
        if not seen[i]:
            cur = []
            dfs(i, cur)
            clusters.append(cur)
    return clusters


# ---------------------------------------------------------------------------
# 4. Stage 2 – dynamic thresholds
# ---------------------------------------------------------------------------

def dyn_muls(boxes):
    h = np.array([b[3] for b in boxes])
    w = np.array([b[2] for b in boxes])
    med_h, med_w = np.median(h), np.median(w)
    iqr_h = np.subtract(*np.percentile(h, [75, 25]))
    iqr_w = np.subtract(*np.percentile(w, [75, 25]))
    a = 1.75
    return a + iqr_h / med_h, (2 - a) * iqr_w / med_w


def close_dyn(b1, b2, vm, hm):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    return (abs((y1 + h1/2) - (y2 + h2/2)) <= vm * min(h1, h2)
            and abs((x1 + w1/2) - (x2 + w2/2)) <= hm * min(w1, w2))


def cluster_dynamic(sub_boxes):
    vm, hm = dyn_muls(sub_boxes)
    N = len(sub_boxes)
    seen = [False] * N
    clusters = []
    def dfs(i, cur):
        seen[i] = True
        cur.append(i)
        for j in range(N):
            if not seen[j] and close_dyn(sub_boxes[i], sub_boxes[j], vm, hm):
                dfs(j, cur)
    for i in range(N):
        if not seen[i]:
            cur = []
            dfs(i, cur)
            clusters.append(cur)
    return clusters

# ---------------------------------------------------------------------------
# 5. Post‑pass fusion 
# ---------------------------------------------------------------------------

def merge_inclusions(boxes, clusters, thr=0.75):
    changed = True
    while changed:
        changed = False
        M = len(clusters)
        i = 0
        while i < M:
            box_i = boxes_from_clusters(boxes, [clusters[i]])[0]
            j = i + 1
            while j < M:
                box_j = boxes_from_clusters(boxes, [clusters[j]])[0]
                if contains(box_i, box_j, thr):
                    # fusion clusters j into i
                    clusters[i].extend(clusters[j])
                    clusters.pop(j)
                    M -= 1
                    changed = True
                elif contains(box_j, box_i, thr):
                    clusters[j].extend(clusters[i])
                    clusters.pop(i)
                    M -= 1
                    changed = True
                    box_i = box_j  # i now points to new merged cluster
                    continue  # restart inner while with same j index
                else:
                    j += 1
            i += 1
    return clusters

# ---------------------------------------------------------------------------
# 6. Pipeline hybride
# ---------------------------------------------------------------------------

def hybrid_blocks_detection(quads, prox_thr=1.0, overlap_thr=0.75,
                            visualize=False, img_path=None):
    boxes = quads_to_rectangles(quads)

    # Stage 1
    base_clusters = cluster_edge(boxes, prox_thr)
    base_boxes    = boxes_from_clusters(boxes, base_clusters)

    # Stage 2 (raffinage)
    refined_clusters = []
    processed = set()
    for i, big in enumerate(base_boxes):
        if i in processed:
            continue
        contained = [i] + [j for j,b in enumerate(base_boxes) if j!=i and contains(big,b,overlap_thr)]
        if len(contained) == 1:
            refined_clusters.append(base_clusters[i])
        else:
            idxs = [k for cid in contained for k in base_clusters[cid]]
            sub_clusters = cluster_dynamic([boxes[k] for k in idxs])
            refined_clusters.extend([[idxs[k] for k in sc] for sc in sub_clusters])
        processed.update(contained)

    # Post‑fusion : merge inclusions restant éventuels
    final_clusters = merge_inclusions(boxes, refined_clusters, overlap_thr)
    final_boxes = boxes_from_clusters(boxes, final_clusters)

    # mapping
    line2cid = [-1] * len(boxes)
    for cid, cl in enumerate(final_clusters):
        for idx in cl:
            line2cid[idx] = cid

            #visualistation
    if visualize and img_path:
        img = mpimg.imread(img_path)
        h, w = img.shape[:2]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        for x, y, bw, bh in final_boxes:
            ax.add_patch(plt.Rectangle((x, y), bw, bh, edgecolor='green', linewidth=1, linestyle='-', fill=False))
        plt.title("Clusters statistiques")
        filename = os.path.basename(img_path)
        print(f"Enregistrement de l'image dans test_new_cluster/{filename}")
        plt.savefig(f"test_new_test/{filename}")

    return final_boxes, line2cid

# ---------------------------------------------------------------------------
# 5.  EXEMPLE 
# ---------------------------------------------------------------------------

# path = "new_test"
# for file in os.listdir(path):
    # file_path = os.path.join(path, file)
    # print(f"Traitement de l'image : {file_path}")
    # quads = apply_craft_detection(file_path, output_dir="outputs")["boxes"]
    # clusters = hybrid_blocks_detection(quads, prox_thr=1.0, overlap_thr=0.75,
                                # visualize=True, img_path=file_path)
    
