import os
import json
import time
import shutil
import re

import fitz  # PyMuPDF
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pytesseract
from surya.layout import LayoutPredictor

# --- Directory configuration (like your pdf_process.py) ---
RAW_PDF_DIR   = "data/raw/pdfs"
PROCESSED_DIR = "data/processed"
VISUAL_DIR    = os.path.join(PROCESSED_DIR, "visual")
BOXES_JSON    = os.path.join(PROCESSED_DIR, "boxes.json")

def empty_cuda_cache():
        """
        Clears the CUDA memory cache if PyTorch is available.
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("CUDA cache cleared.")
        except ImportError:
            print("PyTorch is not installed. Skipping CUDA cache clearing.")

def is_normal_text(text: str) -> bool:
    # disallow control / formatting chars
    if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', text):
        return False
    # disallow invisible / directional / fullwidth ranges
    if re.search(r'[\uFF00-\uFFFF\u2000-\u200F\u202A-\u202E\u2060-\u206F]', text):
        return False
    # allow printable
    return True

# --- Normalize to 0–1000 ---
def normalize_bbox(bbox: list, page_size: list) -> list:
    w, h = page_size
    return [
        int(1000 * bbox[0] / w),
        int(1000 * bbox[1] / h),
        int(1000 * bbox[2] / w),
        int(1000 * bbox[3] / h),
    ]

# --- Initialize Surya & Tesseract ---
layout_predictor = LayoutPredictor()
TESSERACT_CONFIG = "--psm 6"  # single uniform block (paragraph)

# --- Helpers ---

def clean_visualization_dir():
    if os.path.exists(VISUAL_DIR):
        shutil.rmtree(VISUAL_DIR)
    os.makedirs(VISUAL_DIR, exist_ok=True)

def extract_pages_as_images(pdf_path, zoom=1):
    doc = fitz.open(pdf_path)
    pages = []
    for p in doc:
        mat = fitz.Matrix(zoom, zoom)
        pix = p.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append((img, p.rect.width, p.rect.height))
    doc.close()
    return pages

def _visualize_blocks(image, blocks, out_path):
    fig, ax = plt.subplots(1, figsize=(8,10))
    ax.imshow(image)
    for idx, blk in enumerate(blocks, start=1):
        x0,y0,x1,y1 = blk['bbox_px']
        rect = patches.Rectangle((x0,y0), x1-x0, y1-y0,
                                 linewidth=1.5, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x0+5, y0+5, str(idx),
                color='r', fontsize=9, weight='bold',
                bbox=dict(facecolor='white',alpha=0.7,edgecolor='none'))
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved visualization: {out_path}")

# --- Main pipeline ---

def pdf_to_readingbank_surya_ocr(pdf_dir=RAW_PDF_DIR,
                                 output_json=BOXES_JSON,
                                 zoom=2,
                                 visualize=False):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    if visualize:
        clean_visualization_dir()

    entries = []
    pdfs = sorted(f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf'))
    print(f"Found {len(pdfs)} PDFs in {pdf_dir}")
    start = time.time()

    for i, fname in enumerate(pdfs, 1):
        path = os.path.join(pdf_dir, fname)
        print(f"[{i}/{len(pdfs)}] {fname}")
        pages = extract_pages_as_images(path, zoom)

        for page_no, (img, w_pt, h_pt) in enumerate(pages):
            # 1) detect layout boxes
            res = layout_predictor([img])[0]
            boxes = res.bboxes

            # 2) filter paragraphs & char-check
            raw_blocks = []
            for b in boxes:
                lbl = b.label.lower()
                # if lbl in {"formula","picture","figure","equation", "table", "footnote"}:
                    # continue
                # crop & OCR text
                x0,y0,x1,y1 = [int(c) for c in b.bbox]
                raw_blocks.append({
                    'bbox_px': [x0,y0,x1,y1],
                })

            # 3) visualize if needed
            if visualize and raw_blocks:
                vis_path = os.path.join(
                    VISUAL_DIR,
                    f"{os.path.splitext(fname)[0]}_page{page_no+1}.png"
                )
                _visualize_blocks(img, raw_blocks, vis_path)

            # 4) assemble JSON entry
            if raw_blocks:
                page_entry = {
                    'file_name': fname,
                    'page': page_no,
                    'page_size': [w_pt, h_pt],
                    'blocks': []
                }
                for blk in raw_blocks:
                    x0,y0,x1,y1 = blk['bbox_px']
                    # convert px→pt
                    x0_pt, y0_pt, x1_pt, y1_pt = [c/zoom for c in blk['bbox_px']]
                    # normalize 0-1000
                    nb = normalize_bbox(
                        [x0_pt, y0_pt, x1_pt, y1_pt],
                        [w_pt, h_pt]
                    )
                    page_entry['blocks'].append({
                        'bbox': nb
                    })
                entries.append(page_entry)

    # write final JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start
    print(f"\nDone: {len(entries)} pages → {output_json} ({elapsed:.1f}s)")

if __name__ == "__main__":
    import argparse
    empty_cuda_cache()
    p = argparse.ArgumentParser(
        description="PDF→ReadingBank JSON via Surya+Tesseract"
    )
    p.add_argument("--pdf_dir",   default=RAW_PDF_DIR)
    p.add_argument("--output_json", default=BOXES_JSON)
    p.add_argument("--zoom",      type=int, default=1)
    p.add_argument("--visualize", action="store_true")
    args = p.parse_args()

    pdf_to_readingbank_surya_ocr(
        pdf_dir=args.pdf_dir,
        output_json=args.output_json,
        zoom=args.zoom,
        visualize=args.visualize
    )


    
