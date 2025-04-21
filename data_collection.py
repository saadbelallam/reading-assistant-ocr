# data_collection.py
import os
import sys
import json
import cv2
import numpy as np
from src.pipeline.lines_detection1 import apply_craft_detection

# -------------  Utilitaires -------------
def get_image_size(path):
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Impossible de lire {path}")
    h, w = img.shape[:2]
    return w, h

def box_to_dict(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return {"x": min(xs), "y": min(ys), "w": max(xs) - min(xs), "h": max(ys) - min(ys)}

# -------------  Détection CRAFT -------------
def bloc_detection(image_path, output_dir="outputs"):
    pred = apply_craft_detection(image_path, output_dir)
    boxes = pred["boxes"]
    return [b.astype(int).tolist() for b in boxes]

# -------------  GUI pour l’ordre de lecture -------------
ordered_indices, original_boxes, image, dragging = [], [], None, False

def is_in_poly(x, y, poly):
    return cv2.pointPolygonTest(np.array(poly, np.int32), (x, y), False) >= 0

def click_event(ev, x, y, flags, _):
    global dragging, ordered_indices, original_boxes, image
    if ev == cv2.EVENT_LBUTTONDOWN:
        dragging = True
    elif ev == cv2.EVENT_MOUSEMOVE and dragging:
        for i, poly in enumerate(original_boxes):
            if i not in ordered_indices and is_in_poly(x, y, poly):
                ordered_indices.append(i)
                pts = np.array(poly, np.int32)
                cv2.polylines(image, [pts], True, (0, 255, 0), 2)
                c = tuple(np.mean(pts, 0).astype(int))
                cv2.putText(image, str(len(ordered_indices)), c,
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
                cv2.imshow("Image", image)
    elif ev == cv2.EVENT_LBUTTONUP:
        dragging = False
def manual_sort(image_path, polys):
    global ordered_indices, original_boxes, image
    ordered_indices = []
    original_boxes = polys
    image = cv2.imread(image_path)
    if image is None:
        print("Erreur de lecture :", image_path)
        return None

    # dessine les boîtes en rouge
    for i, p in enumerate(polys):
        pts = np.array(p, np.int32)
        cv2.polylines(image, [pts], True, (0, 0, 255), 2)
        c = tuple(np.mean(pts, 0).astype(int))
        cv2.putText(image, str(i + 1), c, cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)

    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", click_event)
    print("Trace l’ordre natural puis :  q = valider  |  c = clear  |  s = skip")

    skip = False
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            break                       # valider
        elif k == ord('c'):
            # reset sélection
            ordered_indices.clear()
            image = cv2.imread(image_path)
            for i, p in enumerate(polys):
                pts = np.array(p, np.int32)
                cv2.polylines(image, [pts], True, (0, 0, 255), 2)
                c = tuple(np.mean(pts, 0).astype(int))
                cv2.putText(image, str(i + 1), c, cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
            cv2.imshow("Image", image)
            print("Sélections effacées.")
        elif k == ord('s'):
            skip = True                 # passer l’image
            break

    cv2.destroyAllWindows()
    if skip:
        print("→ Image skippée.")
        return None

    return {
        "boxes": [box_to_dict(p) for p in polys],
        "reading_order": [i + 1 for i in ordered_indices]
    }
# -------------  Pipeline principal -------------
def update_json(img_dir, json_file="boxes.json"):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    cwd = os.getcwd()

    # --- charge les annotations existantes
    if os.path.exists(json_file):
        try:
            data = json.load(open(json_file, encoding="utf-8"))
        except json.JSONDecodeError:
            print("⚠️  JSON corrompu ; nouveau fichier.")
            data = []
    else:
        data = []

    # dictionnaire nom‑fichier (lower) -> entrée json
    by_name = {os.path.basename(a["image_path"]).lower(): a for a in data}

    # --- scan des images
    for fn in sorted(os.listdir(img_dir)):
        ext = os.path.splitext(fn)[1].lower()
        if ext not in exts:
            continue

        abs_path = os.path.join(img_dir, fn)
        rel_path = os.path.relpath(abs_path, cwd)
        key = fn.lower()  # nom + extension en minuscules

        if key in by_name:
            # déjà annotée → ajouter width/height si manquants
            ann = by_name[key]
            if "width" not in ann or "height" not in ann:
                try:
                    w, h = get_image_size(abs_path)
                    ann["width"], ann["height"] = w, h
                    print(f"✓ taille ajoutée : {fn} ({w}×{h})")
                except Exception as e:
                    print(f"⚠️  impossible d’ouvrir {fn} pour la taille ({e})")
            continue  # pas de GUI

        # --- nouvelle image : détection + GUI
        print(f"\n🆕 {fn}  ▶ détection CRAFT …")
        try:
            polygons = bloc_detection(abs_path)
        except Exception as e:
            print("   CRAFT a échoué :", e)
            continue
        if not polygons:
            print("   Aucun bloc détecté, on saute.")
            continue

        ann = manual_sort(abs_path, polygons)
        if ann is None:
            print("   Annotation annulée.")
            continue

        # ajoute chemin + taille
        try:
            w, h = get_image_size(abs_path)
            ann["width"], ann["height"] = w, h
        except:
            pass
        ann["image_path"] = rel_path
        data.append(ann)
        by_name[key] = ann
        # sauvegarde immédiate
        json.dump(data, open(json_file, "w", encoding="utf-8"), indent=2)
        print(f"   ➕ ajouté ({w}×{h}, {len(ann['boxes'])} boxes)")

    # écriture finale
    json.dump(data, open(json_file, "w", encoding="utf-8"), indent=2)
    print(f"\n✅ Terminé : {len(data)} images annotées au total.")

# -------------  Lancement -------------
def usage():
    print("Usage : python data_collection.py <dossier_images | image_unique>")
    sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        usage()

    target = sys.argv[1]
    if os.path.isdir(target):
        update_json(target)
    else:
        # un seul fichier → on crée un dossier temporaire virtuel
        dir_tmp, file_only = os.path.split(target)
        if not dir_tmp:
            dir_tmp = "."
        update_json(dir_tmp, "boxes.json")
