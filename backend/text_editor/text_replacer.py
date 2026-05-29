import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from backend.utilities import cv_to_pil, pil_to_cv
from backend.text_editor.font_detector import load_matching_font

def erase_text_in_bbox(img_bgr: np.ndarray, bbox: tuple, paper_color_bgr: list) -> np.ndarray:
    """
    Erases text in the specified bounding box by creating a character-level binary mask
    and applying OpenCV Inpainting (Telea) to preserve background texture.
    """
    h_img, w_img = img_bgr.shape[:2]
    x, y, w, h = bbox
    
    # Ensure coordinates are within image bounds
    x = max(0, min(x, w_img - 2))
    y = max(0, min(y, h_img - 2))
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))
    
    # Create a full-image black mask
    full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    
    # Crop local patch
    crop = img_bgr[y:y+h, x:x+w]
    
    try:
        # Convert crop to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Segment character strokes using Otsu's thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Ensure we target the ink strokes rather than the background
        fg_count = np.sum(thresh == 255)
        bg_count = np.sum(thresh == 0)
        if fg_count > bg_count:
            thresh = cv2.bitwise_not(thresh)
            
        # Dilate the character strokes to fully cover complex Devanagari horizontal bars and matras
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_thresh = cv2.dilate(thresh, kernel, iterations=3)
        
        # Inject local mask back into full-image mask
        full_mask[y:y+h, x:x+w] = dilated_thresh
        
    except Exception as e:
        print(f"Error during character mask generation: {e}")
        # Fallback: mask the entire bounding box
        full_mask[y:y+h, x:x+w] = 255
        
    # Apply OpenCV Telea Inpainting to restore background paper textures
    inpainted = cv2.inpaint(img_bgr, full_mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    return inpainted

def replace_text_in_image(
    img_bgr: np.ndarray,
    bbox: tuple,
    replacement_text: str,
    ink_color_bgr: list,
    paper_color_bgr: list,
    font_family: str = "Sans-Serif",
    size_multiplier: float = 0.85
) -> np.ndarray:
    """
    Erases text in a bounding box and renders replacement text with matching style.
    
    Returns:
    OpenCV BGR image containing the replacement text.
    """
    x, y, w, h = bbox
    
    # 1. Erase original text using smart character mask inpainting
    inpainted_bgr = erase_text_in_bbox(img_bgr, bbox, paper_color_bgr)
    
    # 2. Convert to PIL Image for high-quality antialiased text drawing
    pil_img = cv_to_pil(inpainted_bgr)
    draw = ImageDraw.Draw(pil_img)
    
    # 3. Load matching font and scale based on bounding box height (with dynamic character-level font auto-resolver)
    font, font_size = load_matching_font(h, font_family, size_multiplier, text=replacement_text)
    
    # Convert BGR color to RGB for Pillow
    ink_color_rgb = (int(ink_color_bgr[2]), int(ink_color_bgr[1]), int(ink_color_bgr[0]))
    
    # 4. Measure replacement text to align it properly
    # Using modern draw.textbbox or fallback font.getbbox
    try:
        left, top, right, bottom = draw.textbbox((0, 0), replacement_text, font=font)
        text_w = right - left
        text_h = bottom - top
    except AttributeError:
        # Fallback for older PIL versions
        text_w, text_h = font.getsize(replacement_text) if hasattr(font, 'getsize') else (w, h)
        left, top = 0, 0
        
    # Calculate drawing coordinates
    # Center text horizontally or left-align it
    # If text fits inside bounding box, we can left-align or center it
    x_draw = x
    # Center vertically inside the bounding box
    y_draw = y + (h - text_h) // 2 - top
    
    # Draw replacement text
    draw.text((x_draw, y_draw), replacement_text, fill=ink_color_rgb, font=font)
    
    # 5. Convert back to OpenCV BGR
    return pil_to_cv(pil_img)
