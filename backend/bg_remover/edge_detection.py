import cv2
import numpy as np

def guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """
    Fast Guided Filter implementation for edge-preserving matting.
    I: Guidance image (BGR, uint8)
    p: Grayscale mask to filter (uint8 or float, normalized to [0, 1])
    r: Local window radius
    eps: Regularization parameter
    """
    if I.dtype == np.uint8:
        I = I.astype(np.float32) / 255.0
    else:
        I = I.astype(np.float32)
        
    p = p.astype(np.float32)
    
    # Extract grayscale guidance for fast processing
    if len(I.shape) == 3:
        I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    else:
        I_gray = I
        
    # Local window means
    mean_I = cv2.boxFilter(I_gray, -1, (r, r))
    mean_p = cv2.boxFilter(p, -1, (r, r))
    
    mean_II = cv2.boxFilter(I_gray * I_gray, -1, (r, r))
    mean_Ip = cv2.boxFilter(I_gray * p, -1, (r, r))
    
    # Variance & Covariance
    var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    
    # Solve linear coefficients
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    # Average coefficients
    mean_a = cv2.boxFilter(a, -1, (r, r))
    mean_b = cv2.boxFilter(b, -1, (r, r))
    
    # Reconstruct filtered mask
    q = mean_a * I_gray + mean_b
    return np.clip(q, 0.0, 1.0)

def refine_mask(mask: np.ndarray, 
                img: np.ndarray = None,
                closing_size: int = 5, 
                keep_largest_only: bool = True,
                feather_radius: int = 3,
                matting_enabled: bool = False,
                matting_radius: int = 10,
                matting_eps: float = 1e-3) -> np.ndarray:
    """
    Refines a binary mask by morphological closing, speckle filtering, and edge matting/feathering.
    """
    h, w = mask.shape[:2]
    refined = mask.copy()
    
    # 1. Morphological closing to fill holes inside the object
    if closing_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_size, closing_size))
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel) # clean small edge noise
        
    # 2. Keep only the largest connected component to eliminate floating background noise
    if keep_largest_only:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            refined = np.where(labels == largest_label, 255, 0).astype(np.uint8)
            
    # 3. Apply Edge Matting (Guided Filter) or Edge Feathering
    if matting_enabled and img is not None:
        mask_norm = refined.astype(np.float32) / 255.0
        filtered_mask = guided_filter(img, mask_norm, matting_radius, matting_eps)
        refined = (filtered_mask * 255.0).astype(np.uint8)
    elif feather_radius > 0:
        ksize = 2 * feather_radius + 1
        refined_floats = cv2.GaussianBlur(refined.astype(float), (ksize, ksize), 0)
        refined = np.clip(refined_floats, 0, 255).astype(np.uint8)
        
    return refined
