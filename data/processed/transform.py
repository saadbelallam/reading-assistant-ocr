import json
import torch
from transformers import BertTokenizer, BertModel

def transform(input_path, output_path):
    # Initialize BERT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()

    def get_embedding(text):
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", 
                             truncation=True, padding="max_length", 
                             max_length=32).to(device)
            return model(**inputs).last_hidden_state[0, 0].cpu().numpy().tolist()

    with open(input_path, 'r') as f:
        data = json.load(f)

    with open(output_path, 'w') as out_f:
        # First pass: store all boxes and embeddings
        items = {}
        for page in data:
            w, h = page['page_size']
            for idx, block in enumerate(page['blocks']):
                x0, y0, x1, y1 = block['bbox']
                item_id = f"{len(items)}"
                items[item_id] = {
                    "box": [
                        x0 ,  # x
                        y0 ,  # y
                        (x1 - x0) ,  # width
                        (y1 - y0)    # height
                    ],
                    "embedding": get_embedding(block['text'])
                }
                out_f.write(json.dumps({
                    "id": item_id,
                    "box": items[item_id]["box"],
                    "embedding": items[item_id]["embedding"]
                }) + '\n')

        # Second pass: write all pairs
        current_offset = 0
        for page in data:
            n = len(page['blocks'])
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    out_f.write(json.dumps({
                        "box1": str(current_offset + i),
                        "box2": str(current_offset + j),
                        "cls1": str(current_offset + i),
                        "cls2": str(current_offset + j),
                        "label": 1 if j > i else 0
                    }) + '\n')
            current_offset += n

if __name__ == "__main__":
    transform("data/processed/boxes.json", "data/processed/processed.jsonl")