import json


def transform(input_path, output_path):
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
                    ]
                }
                out_f.write(json.dumps({
                    "id": item_id,
                    "box": items[item_id]["box"]
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
                        "label": 1 if j > i else 0
                    }) + '\n')
            current_offset += n

if __name__ == "__main__":
    transform("data/processed/boxes.json", "data/processed/processed.jsonl")