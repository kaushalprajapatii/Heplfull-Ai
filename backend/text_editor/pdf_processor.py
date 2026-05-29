import fitz  # PyMuPDF
from PIL import Image
import io

def get_pdf_page_count(pdf_bytes: bytes) -> int:
    """Return the total number of pages in the PDF bytes."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return len(doc)
    except Exception as e:
        print(f"Error reading PDF page count: {e}")
        return 0

def pdf_page_to_pil(pdf_bytes: bytes, page_index: int, dpi: int = 150) -> Image.Image:
    """Render a PDF page to a high-DPI RGB PIL Image."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_index)
    
    # Scale matrix for high quality DPI rendering (72 is default PDF DPI)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    
    pix = page.get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")
    
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")

def compile_images_to_pdf(pil_images: list) -> bytes:
    """Compile a list of PIL Images into a single PDF bytes object."""
    doc = fitz.open()
    for pil_img in pil_images:
        # Save PIL image as a single-page PDF in a bytes buffer
        pdf_buffer = io.BytesIO()
        pil_img.save(pdf_buffer, format="PDF")
        pdf_buffer.seek(0)
        
        # Open single page PDF and insert into compiled document
        page_doc = fitz.open("pdf", pdf_buffer.read())
        doc.insert_pdf(page_doc)
        
    out_buffer = io.BytesIO()
    doc.save(out_buffer)
    doc.close()
    
    return out_buffer.getvalue()
