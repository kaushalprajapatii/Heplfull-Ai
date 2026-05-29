import cv2
import numpy as np
import math

def blend_rgba(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """
    Perform alpha blending of an RGBA image 'top' over an RGBA image 'bottom'.
    """
    top_a = top[:, :, 3:4].astype(float) / 255.0
    bottom_a = bottom[:, :, 3:4].astype(float) / 255.0
    
    out_a = top_a + bottom_a * (1.0 - top_a)
    out_a_safe = np.where(out_a == 0, 1.0, out_a)
    
    top_rgb = top[:, :, :3].astype(float)
    bottom_rgb = bottom[:, :, :3].astype(float)
    
    out_rgb = (top_rgb * top_a + bottom_rgb * bottom_a * (1.0 - top_a)) / out_a_safe
    
    out_img = np.zeros_like(top)
    out_img[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    out_img[:, :, 3] = np.clip(out_a * 255, 0, 255).astype(np.uint8)[:, :, 0]
    
    return out_img

def generate_drop_shadow(img: np.ndarray, 
                         mask: np.ndarray, 
                         opacity: float = 0.5, 
                         blur_radius: int = 15, 
                         distance: int = 20, 
                         angle_degrees: float = 45.0) -> np.ndarray:
    """
    Generate a realistic drop shadow layer behind the foreground cutout.
    """
    h, w = mask.shape[:2]
    
    cutout = np.zeros((h, w, 4), dtype=np.uint8)
    if img.shape[2] == 4:
        cutout[:, :, :3] = img[:, :, :3]
    else:
        cutout[:, :, :3] = img
    cutout[:, :, 3] = mask
    
    if opacity <= 0.0 or (distance <= 0 and blur_radius <= 0):
        return cutout
        
    shadow = np.zeros((h, w, 4), dtype=np.uint8)
    shadow[:, :, 3] = mask
    
    angle_radians = math.radians(angle_degrees)
    dx = int(distance * math.cos(angle_radians))
    dy = int(distance * math.sin(angle_radians))
    
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    translated_shadow = cv2.warpAffine(
        shadow, M, (w, h), 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0, 0, 0, 0)
    )
    
    if blur_radius > 0:
        ksize = 2 * blur_radius + 1
        translated_shadow = cv2.GaussianBlur(translated_shadow, (ksize, ksize), 0)
        
    translated_shadow[:, :, 3] = (translated_shadow[:, :, 3] * opacity).astype(np.uint8)
    
    final_composite = blend_rgba(top=cutout, bottom=translated_shadow)
    return final_composite
