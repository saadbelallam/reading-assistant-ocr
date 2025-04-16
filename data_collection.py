import cv2
import json
import numpy as np
import sys
import os

from src.pipeline.lines_detection import apply_craft_detection

ordered_indices = []
original_boxes = []
image_data = []
dragging = False
image = None

def bloc_detection(image_path, output_dir):
    prediction = apply_craft_detection(image_path, output_dir)
    craft_boxes = prediction["boxes"]
    polygons = [box.astype(int).tolist() for box in craft_boxes]
    return polygons, list(range(len(polygons)))

def is_point_in_polygon(x, y, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (x, y), False) >= 0

def click_event(event, x, y, flags, param):
    global dragging, ordered_indices, original_boxes, image

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        for idx, poly in enumerate(original_boxes):
            if idx not in ordered_indices and is_point_in_polygon(x, y, poly):
                ordered_indices.append(idx)
                print(f"Polygon {idx + 1} selected (order: {len(ordered_indices)})")

                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(image, [pts], True, (0, 255, 0), 2)
                center = tuple(np.mean(pts, axis=0).astype(int))
                cv2.putText(image, str(len(ordered_indices)), center,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Image', image)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

def box_to_dict(poly):
    x_coords = [pt[0] for pt in poly]
    y_coords = [pt[1] for pt in poly]
    return {
        "x": min(x_coords),
        "y": min(y_coords),
        "w": max(x_coords) - min(x_coords),
        "h": max(y_coords) - min(y_coords)
    }

def manual_box_sorting(image_path, polygons):
    global image, original_boxes, ordered_indices
    original_boxes = polygons
    ordered_indices = []
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None

    for idx, poly in enumerate(polygons):
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(image, [pts], True, (0, 0, 255), 2)
        center = tuple(np.mean(pts, axis=0).astype(int))
        cv2.putText(image, str(idx + 1), center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Image', image)
    cv2.setMouseCallback('Image', click_event)

    print("Drag across the polygons to define reading order. Press 'q' to continue.")
    print("Press 'c' to clear selections and start over.")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            ordered_indices = []
            image = cv2.imread(image_path)
            for idx, poly in enumerate(polygons):
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(image, [pts], True, (0, 0, 255), 2)
                center = tuple(np.mean(pts, axis=0).astype(int))
                cv2.putText(image, str(idx + 1), center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Image', image)
            print("Selections cleared. Start again.")

    cv2.destroyAllWindows()
    annotation = {
        "image_path": image_path,
        "boxes": [box_to_dict(p) for p in polygons],
        "reading_order": [i + 1 for i in ordered_indices]
    }
    return annotation

def main():
    if len(sys.argv) != 2:
        usage()
    path_arg = sys.argv[1]
    output_file = "boxes.json"
    if os.path.isdir(path_arg):
        process_directory(path_arg, output_file)
    else:
        process_single_image(path_arg, output_file)

def process_directory(directory_path, output_file):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if any(f.lower().endswith(ext) for ext in image_extensions)]
    if not image_files:
        print(f"No image files found in {directory_path}")
        return
    existing_data = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading {output_file}, creating new file")
    for image_path in image_files:
        print(f"\nProcessing {image_path}...")
        if any(annotation["image_path"] == image_path for annotation in existing_data):
            print(f"Image {image_path} already processed, skipping")
            continue
        polygons, _ = bloc_detection(image_path, "outputs")
        if not polygons:
            print(f"No blocks detected in {image_path}, skipping")
            continue
        annotation = manual_box_sorting(image_path, polygons)
        if annotation:
            existing_data.append(annotation)
            with open(output_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
            print(f"Annotation added to {output_file} (total: {len(existing_data)} documents)")

def process_single_image(image_path, output_file):
    polygons, _ = bloc_detection(image_path, "outputs")
    if not polygons:
        print(f"No blocks detected in {image_path}")
        return
    annotation = manual_box_sorting(image_path, polygons)
    if annotation:
        existing_data = []
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
        existing_data.append(annotation)
        with open(output_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
        print(f"Annotation added to {output_file} (total: {len(existing_data)} documents)")

def usage():
    print("Usage: python data_collection.py <image_file.jpg or directory_path>")
    sys.exit(1)

if __name__ == "__main__":
    main()
