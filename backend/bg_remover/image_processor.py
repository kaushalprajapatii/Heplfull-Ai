import cv2
import numpy as np
from PIL import Image
from backend.utilities import pil_to_cv, cv_to_pil, resize_for_processing
from backend.bg_remover.segmentation import detect_automatic_bounding_box, run_grabcut
from backend.bg_remover.edge_detection import refine_mask
from backend.bg_remover.shadow_generator import generate_drop_shadow

try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

_session_cache = {}

def get_rembg_session(model_name: str):
    """Retrieves or initializes a cached rembg model session to prevent reloading weights."""
    global _session_cache
    if model_name not in _session_cache:
        try:
            from rembg import new_session
            _session_cache[model_name] = new_session(model_name)
        except Exception as e:
            _session_cache[model_name] = None
    return _session_cache[model_name]

class ImageProcessor:
    """
    High-level orchestrator class to execute the background removal image processing pipeline.
    """
    @staticmethod
    def process_image(
        pil_image: Image.Image,
        rect: tuple = None,
        margin_percentage: float = 5.0,
        iter_count: int = 5,
        bg_seed_sensitivity: float = 35.0,
        closing_size: int = 5,
        keep_largest_only: bool = True,
        feather_radius: int = 3,
        matting_enabled: bool = True,
        matting_radius: int = 10,
        matting_eps: float = 1e-3,
        shadow_enabled: bool = False,
        shadow_opacity: float = 0.5,
        shadow_blur: int = 15,
        shadow_distance: int = 20,
        shadow_angle: float = 45.0,
        max_preview_dim: int = None,
        subject_mode: str = "AI Neural Network (U²-Net)"
    ) -> dict:
        """
        Processes the input PIL image and returns a dictionary of output PIL images.
        """
        # 1. Automatically normalize EXIF camera orientation tags
        from PIL import ImageOps
        pil_image = ImageOps.exif_transpose(pil_image)
        
        # Convert PIL to OpenCV (BGR)
        cv_raw = pil_to_cv(pil_image)
        
        # 2. Downscale for interactive preview speed if requested
        if max_preview_dim is not None:
            cv_img = resize_for_processing(cv_raw, max_preview_dim)
        else:
            cv_img = cv_raw.copy()
            
        h, w = cv_img.shape[:2]
        
        # 3. Bounding Box & Segmentation Determination
        is_neural = (subject_mode in ["AI BiRefNet (SOTA General)", "AI U²-Net (Legacy Neural)", "AI Neural Network (U²-Net)"] and REMBG_AVAILABLE)
        
        if is_neural:
            try:
                # Resolve correct model session
                if "BiRefNet" in subject_mode:
                    session = get_rembg_session("birefnet-general")
                else:
                    session = get_rembg_session("u2net")
                    
                if max_preview_dim is not None:
                    w_p, h_p = cv_img.shape[1], cv_img.shape[0]
                    pil_preview = pil_image.resize((w_p, h_p), Image.Resampling.LANCZOS)
                    cutout_pil = rembg_remove(pil_preview, session=session)
                else:
                    cutout_pil = rembg_remove(pil_image, session=session)
                    
                cv_cutout = pil_to_cv(cutout_pil)
                refined_mask = cv_cutout[:, :, 3].copy()
                
                # Resilient shape normalization to handle EXIF or transpose mismatches from rembg
                if refined_mask.shape[:2] != (h, w):
                    if refined_mask.shape[0] == w and refined_mask.shape[1] == h:
                        refined_mask = refined_mask.T
                    else:
                        refined_mask = cv2.resize(refined_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                        
                actual_rect = (0, 0, w, h)
            except Exception as e:
                is_neural = False
                
        if not is_neural:
            if subject_mode == "Signature & Text (Ink)":
                pw = max(2, min(20, w // 20))
                ph = max(2, min(20, h // 20))
                c_tl = np.mean(cv_img[0:ph, 0:pw, :3], axis=(0, 1))
                c_tr = np.mean(cv_img[0:ph, w-pw:w, :3], axis=(0, 1))
                c_bg = (c_tl + c_tr) / 2.0
                
                dist = np.sqrt(np.sum((cv_img[:, :, :3] - c_bg) ** 2, axis=2))
                
                low_t = 15.0
                high_t = 45.0
                alpha = np.clip((dist - low_t) / (high_t - low_t) * 255.0, 0, 255).astype(np.uint8)
                
                refined_mask = alpha
                if closing_size > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_size, closing_size))
                    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
                    
                actual_rect = (0, 0, w, h)
            else:
                if rect is None:
                    actual_rect = detect_automatic_bounding_box(cv_img, margin_percentage)
                else:
                    if max_preview_dim is not None:
                        orig_h, orig_w = cv_raw.shape[:2]
                        scale_x = w / orig_w
                        scale_y = h / orig_h
                        rx, ry, rw, rh = rect
                        actual_rect = (
                            int(rx * scale_x),
                            int(ry * scale_y),
                            int(rw * scale_x),
                            int(rh * scale_y)
                        )
                    else:
                        actual_rect = rect
                        
                raw_mask = run_grabcut(cv_img, actual_rect, iter_count, bg_seed_sensitivity=bg_seed_sensitivity)
                
                refined_mask = refine_mask(
                    mask=raw_mask, 
                    img=cv_img,
                    closing_size=closing_size, 
                    keep_largest_only=keep_largest_only, 
                    feather_radius=feather_radius,
                    matting_enabled=matting_enabled,
                    matting_radius=matting_radius,
                    matting_eps=matting_eps
                )
        
        # 6. Generate Transparent PNG Cutout
        cutout = np.zeros((h, w, 4), dtype=np.uint8)
        cutout[:, :, :3] = cv_img[:, :, :3]
        cutout[:, :, 3] = refined_mask
        
        # 7. Generate Drop Shadow Composite
        if shadow_enabled:
            shadow_composite = generate_drop_shadow(
                cv_img,
                refined_mask,
                opacity=shadow_opacity,
                blur_radius=shadow_blur,
                distance=shadow_distance,
                angle_degrees=shadow_angle
            )
        else:
            shadow_composite = cutout
            
        # 8. Scale Bounding Box back to original coords if resized (for UI display overlay)
        if max_preview_dim is not None:
            orig_h, orig_w = cv_raw.shape[:2]
            scale_x = orig_w / w
            scale_y = orig_h / h
            ax, ay, aw, ah = actual_rect
            rect_out = (
                int(ax * scale_x),
                int(ay * scale_y),
                int(aw * scale_x),
                int(ah * scale_y)
            )
        else:
            rect_out = actual_rect
            
        # 9. Convert outputs back to PIL
        return {
            "original": cv_to_pil(cv_img),
            "mask": Image.fromarray(refined_mask).convert("L"),
            "transparent": cv_to_pil(cutout),
            "shadow": cv_to_pil(shadow_composite),
            "rect": rect_out
        }
