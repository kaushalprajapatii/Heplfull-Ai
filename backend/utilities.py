import cv2
import numpy as np
from PIL import Image

def pil_to_cv(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL Image to an OpenCV BGR/BGRA numpy array."""
    if pil_image.mode == 'RGBA':
        # RGBA -> BGRA
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)
    else:
        # RGB -> BGR
        return cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)

def cv_to_pil(cv_image: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR/BGRA numpy array to a PIL Image."""
    if len(cv_image.shape) == 3 and cv_image.shape[2] == 4:
        # BGRA -> RGBA
        return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGBA))
    else:
        # BGR -> RGB
        return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

def resize_for_processing(img: np.ndarray, max_dim: int = 800) -> np.ndarray:
    """Resize image preserving aspect ratio if its dimensions exceed max_dim."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    
    if h > w:
        new_h = max_dim
        new_w = int(w * (max_dim / h))
    else:
        new_w = max_dim
        new_h = int(h * (max_dim / w))
        
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def create_checkerboard(width: int, height: int, square_size: int = 16) -> np.ndarray:
    """Create a checkerboard pattern image (RGBA) for transparent preview background."""
    # Create a grid of indices
    y, x = np.indices((height, width))
    # Calculate cell indices
    grid = (x // square_size) + (y // square_size)
    # Binary pattern (0 or 1)
    checkerboard = (grid % 2) * 255
    
    # Map to off-white and gray for a premium look (not harsh black and white)
    # Cell color 1: #F8F9FA (248, 249, 250), Cell color 2: #E9ECEF (233, 236, 239)
    color1 = [248, 249, 250, 255]
    color2 = [233, 236, 239, 255]
    
    bg = np.zeros((height, width, 4), dtype=np.uint8)
    mask = (grid % 2) == 0
    bg[mask] = color1
    bg[~mask] = color2
    
    return bg
