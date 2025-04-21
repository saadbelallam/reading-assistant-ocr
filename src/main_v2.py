#!/usr/bin/env python3
import argparse, os, time, json
import cv2
import shutil
from utils.timing_out import timing_out

from pipeline.lines_detection_v2 import apply_craft_detection
from pipeline.ordering_utils   import reading_order
from pipeline.draw_utils       import draw_polygons, draw_lines
from pipeline.ocr              import extract_text
from pipeline.nlp              import correct_text
from pipeline.audio_generation import generate


def load_cfg():
    with open("config/settings.json") as f:
        return json.load(f)

def prepare_output_folder(img_path, output_dir):
    """
    Crée un sous-dossier propre pour chaque image,
    et copie l'éventuel fichier de vérité 'expected.txt'.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Structure propre : un sous-dossier par image
    rel_path = os.path.relpath(img_path, start="tests/images")
    rel_path = os.path.splitext(rel_path)[0]
    result_path = os.path.join(output_dir, rel_path)
    os.makedirs(result_path, exist_ok=True)

    print(f"Résultats seront stockés dans : {result_path}")

    # Copier expected.txt s'il existe
    base_input, _ = os.path.splitext(img_path)
    expected_txt = base_input + ".txt"
    if os.path.exists(expected_txt):
        dest_txt = os.path.join(result_path, "expected.txt")
        shutil.copy(expected_txt, dest_txt)
        print("Copie de expected.txt dans le dossier de sortie.")
    else:
        print("ℹAucun fichier texte attendu trouvé.")

    return result_path


def process_image(img_path, out_dir, cfg, oro_ckpt,timing ):
    os.makedirs(out_dir, exist_ok=True)
    report, t0 = {}, time.time()
    
    # CRAFT
    if cfg["lines_detection"].get("enabled", False):
        boxes = apply_craft_detection(img_path, out_dir)["boxes"]
        report["CRAFT"] = time.time() - t0; t0 = time.time()


    if cfg["blocks_ordering"].get("enabled", False):
        #parametres
        use_cluster = cfg["ordering"]["use_cluster"]
        algo = cfg["ordering"]["algorithm"]
        decoder = cfg["ordering"].get("oro_decoder", "count")
        # ORO or heatmap (with or without clustering)
        para_ord, line_ord, para_polys = reading_order(
        boxes, img_path,
        oro_ckpt,
        use_cluster=use_cluster,
        decoder=decoder,
        algo=algo,       
        )
        report["ORO"] = time.time() - t0; t0 = time.time()

    # Dessin
    draw_polygons(img_path, [para_polys[i] for i in para_ord],
                  os.path.join(out_dir, "paragraphs.png"))
    draw_lines(img_path, boxes, line_ord,
               os.path.join(out_dir, "lines.png"))
    report["Draw"] = time.time() - t0; t0 = time.time()

    # OCR / NLP / Audio 
    ordered_boxes = [boxes[i] for i in line_ord]
    text, masks = "", []
    if cfg["ocr"]["enabled"]:
        text, masks = extract_text(img_path, out_dir, ordered_boxes, cfg["ocr"])
    if cfg["nlp"]["enabled"]:
        text = correct_text(text, masks, out_dir, cfg["nlp"])
    if cfg["audio_generation"]["enabled"]:
        generate(text, out_dir)
    report["Post"] = time.time() - t0
    if timing: timing_out(report)


if __name__ == "__main__":
    cfg = load_cfg()
    parser = argparse.ArgumentParser(description="Pipeline de traitement de documents.")
    parser.add_argument("--input", required=True, help="Chemin de l'image à traiter")
    parser.add_argument("--output-dir", required=True, help="Dossier où sauvegarder les résultats")
    parser.add_argument("-t", "--timing", action="store_true", help="Afficher les temps d'exécution")
    args = parser.parse_args()

    # Préparer proprement le dossier résultat pour cette image
    result_path = prepare_output_folder(args.input, args.output_dir)

    # Lancer le pipeline
    process_image(
        img_path=args.input,
        out_dir=result_path,
        cfg=cfg,
        oro_ckpt="models/Order_Relation_Operator/model.pth",
        timing=args.timing,
    )
