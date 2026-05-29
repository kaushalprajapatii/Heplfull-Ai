import cv2
import numpy as np

def detect_automatic_bounding_box(img: np.ndarray, margin_percentage: float = 5.0) -> tuple:
    """
    Detect the main subject's bounding box using edge density and contour analysis.
    Filters out full-frame border contours.
    """
    h, w = img.shape[:2]
    
    # 1. Preprocess: Convert to grayscale and blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Compute edge density using Canny
    edges = cv2.Canny(blurred, 30, 100)
    
    # 3. Morphological closing to join close edge segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(closed, kernel, iterations=2)
    
    # 4. Find contours of the edge-dense regions
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If contours exist, find the bounding box of the largest ones
    if contours:
        # Filter contours by area and ignore frame border contours that cover the entire image
        valid_contours = []
        for c in contours:
            cx, cy, ccw, cch = cv2.boundingRect(c)
            if ccw >= w - 10 and cch >= h - 10:
                continue
            if cv2.contourArea(c) > (w * h * 0.002):
                valid_contours.append(c)
                
        # Sort valid contours by area descending
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        
        if valid_contours:
            # Get the unified bounding box of the significant contours
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for c in valid_contours[:3]:  # Top 3 largest contours
                x, y, cw, ch = cv2.boundingRect(c)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + cw)
                y_max = max(y_max, y + ch)
            
            # Add a slight padding/margin
            pad_x = int((x_max - x_min) * 0.05)
            pad_y = int((y_max - y_min) * 0.05)
            
            x = max(0, x_min - pad_x)
            y = max(0, y_min - pad_y)
            cw = min(w - x, (x_max - x_min) + 2 * pad_x)
            ch = min(h - y, (y_max - y_min) + 2 * pad_y)
            
            # Ensure it is valid
            if cw > 10 and ch > 10:
                return (x, y, cw, ch)
                
    # Fallback: Center-based bounding box with specified margin
    margin_w = int(w * (margin_percentage / 100.0))
    margin_h = int(h * (margin_percentage / 100.0))
    
    x = margin_w
    y = margin_h
    cw = w - (2 * margin_w)
    ch = h - (2 * margin_h)
    
    return (x, y, cw, ch)

def run_grabcut(img: np.ndarray, rect: tuple, iter_count: int = 5, bg_seed_sensitivity: float = 35.0) -> np.ndarray:
    """
    Run GrabCut algorithm on BGR image using the provided bounding box.
    Optionally initializes the GrabCut mask with background seeds from the top corners.
    """
    h, w = img.shape[:2]
    
    # Ensure rect is valid and within image boundaries
    rx, ry, rw, rh = rect
    rx = max(0, min(rx, w - 2))
    ry = max(0, min(ry, h - 2))
    rw = max(1, min(rw, w - rx))
    rh = max(1, min(rh, h - ry))
    safe_rect = (rx, ry, rw, rh)
    
    # Initialize GrabCut background/foreground models
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    
    # Create GrabCut mask
    mask = np.zeros((h, w), dtype=np.uint8)
    
    try:
        if bg_seed_sensitivity > 0.0 and len(img.shape) == 3:
            # 1. Initialize entire mask inside the bounding box as Probable Foreground (3)
            # and outside as Sure Background (0)
            cv2.rectangle(mask, (rx, ry), (rx + rw, ry + rh), cv2.GC_PR_FGD, -1)
            
            # 2. Sample only top corner regions (Top-Left and Top-Right) to detect background color
            pw = max(2, min(20, w // 20))
            ph = max(2, min(20, h // 20))
            
            # Extract corner colors (BGR channels average)
            c_tl = np.mean(img[0:ph, 0:pw, :3], axis=(0, 1))
            c_tr = np.mean(img[0:ph, w-pw:w, :3], axis=(0, 1))
            
            # 3. For each pixel, compute distance to nearest corner color
            diff_tl = np.sqrt(np.sum((img[:, :, :3] - c_tl) ** 2, axis=2))
            diff_tr = np.sqrt(np.sum((img[:, :, :3] - c_tr) ** 2, axis=2))
            
            min_diff = np.minimum(diff_tl, diff_tr)
            
            # 4. Mark pixels inside safe_rect that are very close to corner colors as Probable Background (2)
            bg_mask = (min_diff < bg_seed_sensitivity)
            bbox_mask = np.zeros((h, w), dtype=bool)
            bbox_mask[ry:ry+rh, rx:rx+rw] = True
            
            # Final probable background assignments
            mask[bbox_mask & bg_mask] = cv2.GC_PR_BGD
            
            # Run GrabCut in MASK mode
            cv2.grabCut(img, mask, safe_rect, bgd_model, fgd_model, iter_count, cv2.GC_INIT_WITH_MASK)
        else:
            # Traditional RECT-only initialization
            cv2.grabCut(img, mask, safe_rect, bgd_model, fgd_model, iter_count, cv2.GC_INIT_WITH_RECT)
        
        # Convert output mask to binary
        binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        return binary_mask
    except Exception as e:
        fallback_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(fallback_mask, (rx, ry), (rx + rw, ry + rh), 255, -1)
        return fallback_mask
