import numpy as np

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

_reader = None

def get_ocr_reader():
    """Lazy initializer for EasyOCR reader."""
    global _reader
    if not EASYOCR_AVAILABLE:
        return None
        
    if _reader is None:
        try:
            # Initialize with English and Hindi, CPU only for maximum compatibility
            _reader = easyocr.Reader(['hi', 'en'], gpu=False)
        except Exception as e:
            print(f"Error initializing EasyOCR: {e}")
            return None
    return _reader

def detect_text_in_image(img_bgr: np.ndarray) -> list:
    """
    Detect all text strings and their bounding boxes in the BGR image.
    
    Returns:
    A list of dictionaries:
    [
        {
            "text": str,
            "bbox": (x, y, w, h),
            "confidence": float
        }
    ]
    """
    reader = get_ocr_reader()
    if reader is None:
        print("OCR Engine not available. Skipping automatic text detection.")
        return []
        
    try:
        # EasyOCR requires RGB or BGR numpy array
        # It returns: [([[x0,y0], [x1,y1], [x2,y2], [x3,y3]], text, confidence), ...]
        raw_results = reader.readtext(img_bgr)
        
        processed_results = []
        for coords, text, confidence in raw_results:
            # Map four-point coordinates to standard (x, y, w, h) bounding box
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            
            x = int(min(xs))
            y = int(min(ys))
            w = int(max(xs) - x)
            h = int(max(ys) - y)
            
            processed_results.append({
                "text": text.strip(),
                "bbox": (x, y, w, h),
                "confidence": float(confidence)
            })
            
        return processed_results
    except Exception as e:
        print(f"OCR detection encountered an error: {e}")
        return []
