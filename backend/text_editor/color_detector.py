import cv2
import numpy as np

def detect_ink_and_paper_colors(img_bgr: np.ndarray, bbox: tuple) -> tuple:
    """
    Analyzes a local bounding box region to extract the BGR colors of:
    1. The ink (foreground text)
    2. The paper (background backdrop)
    
    Parameters:
    - img_bgr: Source BGR image
    - bbox: Bounding box coordinates (x, y, w, h)
    
    Returns:
    A tuple (ink_color_bgr, paper_color_bgr) as lists of 3 integers.
    """
    h_img, w_img = img_bgr.shape[:2]
    x, y, w, h = bbox
    
    # Ensure coordinates are within image bounds
    x = max(0, min(x, w_img - 2))
    y = max(0, min(y, h_img - 2))
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))
    
    # Crop local patch
    crop = img_bgr[y:y+h, x:x+w]
    
    # Defaults
    default_ink = [0, 0, 0]      # Black
    default_paper = [255, 255, 255]  # White
    
    if crop.size == 0:
        return default_ink, default_paper
        
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Apply Otsu's thresholding to separate ink from paper
        # If paper is bright and ink is dark, THRESH_BINARY_INV makes ink pixels 255, paper pixels 0
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Count foreground vs background pixel coverage
        fg_count = np.sum(thresh == 255)
        bg_count = np.sum(thresh == 0)
        
        # If OTSU thresholding binarized incorrectly (e.g. inverted because text is white on black background)
        # We assume the background (paper) takes up the majority of the crop area
        if fg_count > bg_count:
            # Invert the mask
            thresh = cv2.bitwise_not(thresh)
            fg_count, bg_count = bg_count, fg_count
            
        ink_pixels = crop[thresh == 255]
        paper_pixels = crop[thresh == 0]
        
        if len(ink_pixels) > 0:
            # Use median to avoid outliers (like gradient noise near text borders)
            ink_color = np.median(ink_pixels, axis=0).astype(int).tolist()
        else:
            ink_color = default_ink
            
        if len(paper_pixels) > 0:
            paper_color = np.median(paper_pixels, axis=0).astype(int).tolist()
        else:
            paper_color = default_paper
            
        return ink_color, paper_color
    except Exception as e:
        print(f"Error extracting colors from bounding box: {e}")
        return default_ink, default_paper
