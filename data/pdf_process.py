# pdf_process.py
# Modified to filter out blocks with weird characters

import os
import json
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import time
import shutil
import re  # For character filtering

# Directory configuration
RAW_PDF_DIR = "data/raw/pdfs"
PROCESSED_DIR = "data/processed"
VISUAL_DIR = os.path.join(PROCESSED_DIR, "visual")
BOXES_JSON = os.path.join(PROCESSED_DIR, "boxes.json")

# Define what constitutes "normal" characters
NORMAL_CHAR_REGEX = re.compile(
    r'^[\w\s'  # Alphanumeric and whitespace
    r'\.\,\;\:\'\"\!\?\@\#\$\%\&\*\(\)\[\]\{\}\-\+\=\/\|\\'  # Common punctuation
    r']+$', 
    flags=re.UNICODE
)

def is_normal_text(text: str) -> bool:
    """More permissive check that allows academic/technical text"""
    # First check for obviously invalid patterns
    if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', text):  # Control chars
        return False
        
    # Then allow most printable characters except specific unwanted ones
    return not re.search(
        r'['
        r'\uFF00-\uFFFF'  # Full-width/half-width chars
        r'\u2000-\u200F'  # Special spaces
        r'\u202A-\u202E'  # Directional formatting
        r'\u2060-\u206F'  # Invisible formatting
        r']',
        text,
        flags=re.UNICODE
    )

def clean_visualization_dir(output_dir: str) -> None:
    """Clear all files in the visualization directory"""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

def extract_text_blocks_from_pdf(pdf_path: str) -> list:
    """Extract text blocks while filtering weird characters"""
    doc = fitz.open(pdf_path)
    blocks = []
    
    for page_no in range(len(doc)):
        page = doc[page_no]
        width, height = page.rect.width, page.rect.height
        
        page_dict = page.get_text("dict")
        
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # Skip non-text blocks
                continue
                
            block_text = []
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        block_text.append(text)
            
            if not block_text:
                continue
                
            full_text = "\n".join(block_text)
            
            # Skip blocks with weird characters
            if not is_normal_text(full_text):
                continue
                
            x0, y0, x1, y1 = block["bbox"]
            
            blocks.append({
                'text': full_text,
                'bbox': [x0, y0, x1, y1],
                'page': page_no,
                'page_size': [width, height]
            })
    
    return blocks

def normalize_bbox(bbox: list, page_size: list) -> list:
    """Normalize bbox coordinates to 0-1000 range."""
    width, height = page_size
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / height),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / height),
    ]

def visualize_pdf_blocks(pdf_path: str, 
                       page_no: int = 0, 
                       output_dir: str = VISUAL_DIR) -> str:
    """Visualize text blocks with numbered labels"""
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{pdf_basename}_page{page_no}_viz.png")
    
    try:
        all_blocks = extract_text_blocks_from_pdf(pdf_path)
        page_blocks = [b for b in all_blocks if b['page'] == page_no]
        
        if not page_blocks:
            print(f"No valid blocks found on page {page_no} of {pdf_path}")
            return None
        
        page_width, page_height = page_blocks[0]['page_size']
        
        # Render PDF page
        doc = fitz.open(pdf_path)
        if page_no >= len(doc):
            print(f"Page {page_no} does not exist in {pdf_path}")
            return None
            
        page = doc[page_no]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=(12, 16))
        ax.imshow(np.array(img))
        
        # Draw bounding boxes with incremental numbers
        for idx, block in enumerate(page_blocks, 1):
            x0, y0, x1, y1 = block['bbox']
            
            # Scale coordinates
            x0 = x0 * pix.width / page_width
            y0 = y0 * pix.height / page_height
            x1 = x1 * pix.width / page_width
            y1 = y1 * pix.height / page_height
            
            width = x1 - x0
            height = y1 - y0
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x0, y0), width, height, 
                linewidth=1.5, edgecolor='r', facecolor='none', alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add number label
            ax.text(
                x0 + 5, y0 + 15, str(idx),
                color='red', fontsize=10, weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
            
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"PDF: {pdf_basename}, Page: {page_no+1}")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Visualization saved: {os.path.basename(output_path)}")
        return output_path
        
    except Exception as e:
        print(f"Error visualizing {pdf_path}, page {page_no}: {str(e)}")
        return None

def pdf_to_readingbank_json(pdf_dir: str = RAW_PDF_DIR, 
                          output_json: str = BOXES_JSON, 
                          process_all_pages: bool = True) -> None:
    """Process PDFs and save filtered text blocks to JSON"""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(VISUAL_DIR, exist_ok=True)
    
    json_entries = []
    pdf_files = [f for f in sorted(os.listdir(pdf_dir)) if f.lower().endswith('.pdf')]
    total_files = len(pdf_files)
    
    print(f"Found {total_files} PDF files in {pdf_dir}")
    print(f"Processing PDF files{'(all pages)' if process_all_pages else '(first page only)'}")
    print("Filtering blocks with non-standard characters...")
    
    for file_idx, fname in enumerate(pdf_files, 1):
        path = os.path.join(pdf_dir, fname)
        print(f"[{file_idx}/{total_files}] Processing {fname}...", end="", flush=True)
        
        try:
            blocks = extract_text_blocks_from_pdf(path)
            if not blocks:
                print(" No valid text found.")
                continue
            
            doc = fitz.open(path)
            total_doc_pages = len(doc)
            doc.close()
            
            pages = {}
            for block in blocks:
                page_no = block['page']
                if not process_all_pages and page_no > 0:
                    continue
                    
                entry = pages.setdefault(page_no, {
                    'file_name': fname,
                    'page': page_no,
                    'page_size': block['page_size'],
                    'blocks': []
                })
                
                nbbox = normalize_bbox(block['bbox'], block['page_size'])
                entry['blocks'].append({
                    'text': block['text'],
                    'bbox': nbbox
                })
            
            processed_pages = len(pages)
            for page_entry in pages.values():
                json_entries.append(page_entry)
            
            if process_all_pages:
                print(f" Processed {processed_pages}/{total_doc_pages} pages with valid text.")
            else:
                print(" Processed first page.")
                
            if file_idx % 10 == 0 or file_idx == total_files:
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(json_entries, f, ensure_ascii=False, indent=2)
                    
        except Exception as e:
            print(f" Error: {str(e)}")
            
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_entries, f, ensure_ascii=False, indent=2)
    
    print(f"Processing complete! Results saved to {output_json}")

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='PDF to ReadingBank JSON converter')
    parser.add_argument('--pdf_dir', default=RAW_PDF_DIR, help='PDF directory')
    parser.add_argument('--output_json', default=BOXES_JSON, help='Output JSON path')
    parser.add_argument('--all_pages', action='store_true', help='Process all pages')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--viz_dir', default=VISUAL_DIR, help='Visualization directory')
    parser.add_argument('--viz_page', type=int, help='Specific page to visualize')
    parser.add_argument('--max_files', type=int, help='Limit number of files to process')
    args = parser.parse_args()
    
    if not os.path.isdir(args.pdf_dir):
        print(f"Error: PDF directory {args.pdf_dir} does not exist")
        exit(1)
    
    print("=" * 80)
    print(f"PDF Processing Tool - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    pdf_to_readingbank_json(args.pdf_dir, args.output_json, args.all_pages)
    
    if args.visualize:
        print("\nGenerating visualizations:")
        clean_visualization_dir(args.viz_dir)
        print(f"Cleared visualization directory: {args.viz_dir}")
        
        pdf_files = [f for f in sorted(os.listdir(args.pdf_dir)) if f.lower().endswith('.pdf')]
        if args.max_files:
            pdf_files = pdf_files[:args.max_files]
            
        for file_idx, fname in enumerate(pdf_files, 1):
            pdf_path = os.path.join(args.pdf_dir, fname)
            print(f"[{file_idx}/{len(pdf_files)}] Visualizing {fname}...")
            
            try:
                if args.viz_page is not None:
                    visualize_pdf_blocks(pdf_path, args.viz_page, args.viz_dir)
                else:
                    doc = fitz.open(pdf_path)
                    num_pages = len(doc)
                    doc.close()
                    pages_to_visualize = range(num_pages) if args.all_pages else [0]
                    for page_no in pages_to_visualize:
                        visualize_pdf_blocks(pdf_path, page_no, args.viz_dir)
            except Exception as e:
                print(f"  Error: {str(e)}")
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    import argparse
    main()