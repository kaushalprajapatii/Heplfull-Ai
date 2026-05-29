# ⚡ Antigravity Studio - AI Image & Document Workspace

A complete, production-grade, state-of-the-art **Image & PDF Manipulation Web Application** consisting of two primary workspaces:
1. **AI Background Remover Workspace**: Isolate foreground subjects with U²-Net deep learning or GrabCut, refine fine hair details with Guided Filter matting, and generate customizable realistic drop shadows.
2. **AI In-Image Text Editor Workspace**: Automatically detect, erase, and replace text in images or multi-page PDFs while perfectly matching original fonts, sizes, and ink colors.

Built with **Streamlit** (frontend) and **Python + OpenCV + Deep Learning** (backend).

---

## ✨ Workspace 1: AI Background Remover

Allows users to upload an image, extract the subject with professional-grade borders, and customize visual compositions.

- **AI U²-Net Neural Network Mode**: Leverages pre-trained deep learning to automatically segment complex shapes (like portraits, hair, and soft details) without bounding box configurations.
- **GrabCut General Subject Mode**: Detects bounding boxes automatically or supports manual coordinate tuning.
- **Signature & Text (Ink Extraction) Mode**: Uses color distance maps to isolate handwritten ink from paper backdrops (perfect for signatures).
- **Advanced Edge Matting**:
  - *Guided Filter Matting*: An edge-preserving linear filtering algorithm that aligns alpha transitions to individual hair fibers.
  - *Morphological Closing*: Fills inner holes and smooths contours.
  - *Speckle Filtering*: Cleans isolated floating noise.
- **Realistic Drop Shadow Engine**: Generates natural drop shadows beneath isolated subjects. Customize opacity, blur softness, offset distance, and angle.
- **Interactive Composite Studio**: Composite transparent cutouts over Checkerboard grids, Solid colors, gradients (*Midnight Glow, Sunset Studio, Neon Cyber*), or uploaded background images.

---

## ✨ Workspace 2: AI Document & PDF Text Editor

Allows users to upload an image or multi-page PDF, find text, and perform clean in-place corrections.

- **Automated Layout Scan**: Scans images or document page textures using **EasyOCR** to discover bounding boxes of all text segments.
- **Translucent OCR Overlays**: Renders glowing overlays displaying editable text fields in the document.
- **Zero-Configuration PDF Engine**: Powered by **PyMuPDF (Fitz)**. Render multi-page PDFs, navigate pages, and compile edited page images back into a single multi-page PDF document.
- **Adaptive Ink & Paper Color Sampling**: Extracts the median BGR ink color (foreground character strokes) and surrounding paper texture color (background) using Otsu's adaptive binarization of characters.
- **Seamless Texture-Preserving Eraser**: Creates a character-level binary mask, dilates it by 2px, and runs OpenCV Fast Inpainting (**Telea's algorithm**). This erases old text while preserving paper folds, grain, gridlines, and shadows.
- **Adaptive Text Synthesis**: Scales the replacement text according to the bounding box height, matches standard font families (Serif, Sans-Serif, Monospace), and renders it anti-aliased with the sampled ink color.

---

## 📂 Project Structure

```text
e:\bg remover\
├── app.py                     # Main Streamlit dashboard routing entrypoint & CSS styles
├── requirements.txt           # Unified dependency manifest
├── README.md                  # Detailed workspace and setup documentation
│
├── frontend/                  # Modern UI Layout Modules
│   ├── __init__.py
│   ├── home.py                # Premium Glassmorphic Studio Navigation landing page
│   ├── bg_remover_ui.py       # Custom Background Remover controls and workflows
│   ├── text_editor_ui.py      # PDF page navigation, OCR list, Find/Replace editor
│   ├── preview_ui.py          # Before/After comparison and backdrop composites
│   └── download_ui.py         # Downscaled/lazy high-res download compiler
│
└── backend/                   # Algorithms and Heavy Processing Backends
    ├── __init__.py
    ├── utilities.py           # Image format converters and checkerboard pattern generators
    │
    ├── bg_remover/            # Background Remover Subpackage
    │   ├── __init__.py
    │   ├── segmentation.py    # GrabCut, contour analysis, and corner color GMM seeding
    │   ├── edge_detection.py  # Fast Guided Filter matting and morphological refinements
    │   ├── shadow_generator.py# Affine transformation drop shadow translations and blenders
    │   └── image_processor.py # Background remover high-level pipeline orchestrator
    │
    └── text_editor/           # Text Editor & Document Subpackage
        ├── __init__.py
        ├── pdf_processor.py   # PyMuPDF page extractors and PDF compilers
        ├── ocr_engine.py      # EasyOCR layout scanning engine
        ├── color_detector.py  # Character-level ink/paper color samplers
        ├── font_detector.py   # System TTF font loader and height-based size mapper
        ├── text_replacer.py   # Telea inpainter and PIL anti-aliased text renderer
        └── orchestrator.py    # Text correction orchestrator
```

---

## 🚀 Installation & Running Locally

Ensure you have **Python 3.10+** installed. Follow these steps:

### 1. Install Dependencies
Open a terminal in the root of the workspace (`e:\bg remover\`) and run:
```bash
pip install -r requirements.txt
```

### 2. Run the Application
Start the Streamlit dashboard:
```bash
streamlit run app.py
```

### 3. Open in Browser
The local server will start and prompt you to open the dashboard at:
- **Local URL**: `http://localhost:8501`

---

## 🛠️ Offline Verification & Test Suites

We have built extensive automated tests to guarantee system stability and code correctness:

- **Background Remover Tests**:
  ```bash
  python "C:\Users\Asus\.gemini\antigravity\brain\76a54bfc-09a7-4a96-809d-8b8c797bd925\scratch\test_pipeline.py"
  ```
- **Document Text Editor Tests**:
  ```bash
  python "C:\Users\Asus\.gemini\antigravity\brain\76a54bfc-09a7-4a96-809d-8b8c797bd925\scratch\test_text_editor.py"
  ```
Both test suites should print `[SUCCESS]` and pass seamlessly!
