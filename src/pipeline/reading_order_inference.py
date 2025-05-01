"""
Fast Optimization for Reading Order Inference
"""

import torch
import numpy as np
import math
import itertools
from transformers import BertTokenizer, BertModel
from models.reading_order.model import ReadingOrderModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
bert_model.eval()

# Embedding function
def get_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=32).to(device)
        return bert_model(**inputs).last_hidden_state[0, 0].cpu()

# Load trained model
def load_reading_order_model(path="models/reading_order/best_model.pt"):
    model = ReadingOrderModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

model = load_reading_order_model()

# Predict one probability: p(bi → bj)
def predict_parent_probability(bi, bj, text_i, text_j):
    box1 = torch.tensor(bi, dtype=torch.float32).to(device)
    box2 = torch.tensor(bj, dtype=torch.float32).to(device)
    cls1 = get_embedding(text_i).to(device)
    cls2 = get_embedding(text_j).to(device)

    with torch.no_grad():
        output = model(box1.unsqueeze(0), box2.unsqueeze(0), cls1.unsqueeze(0), cls2.unsqueeze(0))
        prob = torch.sigmoid(output).item()
    return prob

# Predict full matrix of p(bi → bj)
def predict_parent_matrix(blocks, texts):
    n = len(blocks)
    embeddings = [get_embedding(t).to(device) for t in texts]
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            box1 = torch.tensor(blocks[i], dtype=torch.float32).to(device)
            box2 = torch.tensor(blocks[j], dtype=torch.float32).to(device)
            cls1 = embeddings[i]
            cls2 = embeddings[j]

            with torch.no_grad():
                prob = torch.sigmoid(model(box1.unsqueeze(0), box2.unsqueeze(0), cls1.unsqueeze(0), cls2.unsqueeze(0))).item()
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

if __name__ == "__main__":
    # Example usage
    blocks = [
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 15]
    ]
    texts = ["Hello", "Test", "World"]
    
    # Predict parent matrix
    P = predict_parent_matrix(blocks, texts)
    print("Parent matrix P:", P)
    
    # Decode using Held-Karp algorithm
    order = held_karp_decoder(P)
    
    print("Predicted reading order:", order)