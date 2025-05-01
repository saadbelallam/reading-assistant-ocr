import json
import torch
from torch.utils.data import Dataset

class ReadingOrderDataset(Dataset):
    def __init__(self, jsonl_path):
        self.pairs = []
        self.boxes = {}
        self.embeddings = {}

        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if "id" in entry:
                    # Definition line
                    self.boxes[entry["id"]] = entry["box"]
                    self.embeddings[entry["id"]] = entry["embedding"]
                else:
                    # Pair line
                    self.pairs.append({
                        "box1": entry["box1"],
                        "box2": entry["box2"],
                        "cls1": entry["cls1"],
                        "cls2": entry["cls2"],
                        "label": entry["label"]
                    })

    def __len__(self):
        return len(self.pairs)

    #def __getitem__(self, idx):
    #    pair = self.pairs[idx]
#
    #    return {
    #        "box1": self.boxes[pair["box1"]],
    #        "box2": self.boxes[pair["box2"]],
    #        "cls1": self.embeddings[pair["cls1"]],
    #        "cls2": self.embeddings[pair["cls2"]],
    #        "label": pair["label"]
    #    }
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return {
            "box1": torch.tensor(self.boxes[pair["box1"]], dtype=torch.float32),
            "box2": torch.tensor(self.boxes[pair["box2"]], dtype=torch.float32),
            "cls1": torch.tensor(self.embeddings[pair["cls1"]], dtype=torch.float32),
            "cls2": torch.tensor(self.embeddings[pair["cls2"]], dtype=torch.float32),
            "label": torch.tensor(pair["label"], dtype=torch.float32),
        }