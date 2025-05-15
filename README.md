# Reading assistant for visually impaired people

Point a camera at a package, a sign or a magazine page, and the app reads the text out loud.

## Why

Regular OCR works well on clean scans. It struggles with what people actually need to read day to day: curved packaging, bad lighting, tilted text, multi-column layouts. It also reads everything top to bottom, which breaks as soon as the page isn't a single column. We focused on these two problems: getting the text right in messy photos, and reading it in the right order.

## Pipeline

```
image
  → text line detection (CRAFT)
  → grouping lines into paragraphs
  → deskew + geometric rectification
  → reading order
  → OCR (Tesseract)
  → text correction (CamemBERT)
  → speech (gTTS)
```

A few parts worth looking at:

**Paragraph clustering.** K-means and GMM need the number of clusters up front, and HDBSCAN kept labelling valid lines as noise. So we wrote our own. Line boxes are linked when their distance, normalised by the median line height, is under a threshold, and a DFS gives the connected components. Clusters that swallowed other ones are then re-split with thresholds based on the IQR of line heights and widths, so the algorithm adapts to each page.

**Reading order.** We tried two approaches. A heuristic that moves a directional "attention cone" across the page, and a small MLP (under 30k parameters) that predicts, for each pair of blocks, which one comes first. The global order is then decoded from the pairwise probabilities (brute force, Held-Karp or greedy depending on the number of blocks). The MLP was trained on block orders extracted with Surya OCR from arXiv papers, books and magazines.

**OCR correction.** Low-confidence words from Tesseract are replaced with a `<mask>` token, CamemBERT proposes candidates, and we keep the one closest to what the OCR actually saw (Levenshtein / fuzzy matching).

## Results

We couldn't find a public French dataset with ground truth, so we built two: 100 phone photos taken in real conditions, and 131 scanned magazine pages.

Character accuracy:

|                  | Full pipeline | CRAFT + Tesseract | Tesseract only |
|------------------|---------------|-------------------|----------------|
| Real photos      | **64.1%**     | 49.3%             | 51.0%          |
| Scanned pages    | 67.5%         | 36.1%             | **70.2%**      |

On clean scans, plain Tesseract is slightly better and the extra steps don't help. On real photos, the full pipeline wins clearly. These numbers also ignore reading order, where Tesseract alone just goes top to bottom.

Reading order accuracy (τ, 1 = perfect order):

| Pairwise MLP | Heatmap heuristic | XY-Cut |
|--------------|-------------------|--------|
| **0.86**     | 0.64              | 0.69   |

## Limitations

Heavily crumpled documents and steep angles still fail. Low contrast hurts Tesseract a lot. The whole chain is too slow for live video without optimisation, and gTTS adds its own latency.

## Getting started

You'll need [uv](https://docs.astral.sh/uv/) and Tesseract with the French language data (`tesseract-ocr-fra` on Ubuntu, `brew install tesseract tesseract-lang` on macOS).

```bash
make install
```

This creates a `.venv/`, installs the dependencies from `pyproject.toml`, and offers to download CamemBERT (about 550 MB).

Then:

```bash
make webcam                              # take a photo with your webcam and read it
make process-image image=path/to/img.jpg # run the pipeline on one image
make process-all                         # run it on everything in tests/images/
```

Add `time=true` to either `process` command to log execution time and RAM usage.

To evaluate the OCR output against ground truth:

```bash
make evaluate image=path/to/img.jpg
make evaluate-all                        # everything in tests/results/
```

`make clean` removes generated results, `make delete` also removes the venv and the CamemBERT model. `make help` lists everything.

## Stack

Python 3.10, PyTorch, OpenCV, Transformers (CamemBERT), PyTesseract, Shapely, HDBSCAN, gTTS, uv, Make.
