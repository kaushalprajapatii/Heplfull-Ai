import cv2
import numpy as np
from PIL import Image
from backend.utilities import pil_to_cv, cv_to_pil
from backend.bg_remover.image_processor import ImageProcessor
from backend.dslr_blur.blur_processor import DSLRBlurProcessor
from backend.enhancement.onnx_engine import ONNXInferenceEngine

class DepthBlurEngine:
    """
    Fuses Depth Anything V2 depth maps and BiRefNet masks to execute 
    spatially-varying, optical DSLR depth-of-field blurs.
    """
    
    @classmethod
    def generate_synthetic_depth_map(cls, h: int, w: int, mask: np.ndarray) -> np.ndarray:
        """
        Fallback generator that constructs a smooth vertical linear perspective depth map
        in case the neural ONNX model is offline or unavailable.
        """
        # Create vertical gradient: 0 at top (infinity), 255 at bottom (foreground ground plane)
        y, x = np.indices((h, w))
        depth = (y / h * 255.0).astype(np.uint8)
        
        # Clamp subject pixels to foreground (240) to keep the focus plane sharp
        depth[mask > 127] = 230
        return depth

    @classmethod
    def run_depth_estimation(cls, cv_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Attempts to run Depth Anything V2 neural depth estimation with a graceful 
        vertical linear perspective fallback on network/load failures.
        """
        h, w = cv_img.shape[:2]
        try:
            # Try running ONNX Depth Anything session
            depth_map = ONNXInferenceEngine.run_depth_anything(cv_img[:, :, :3])
            return depth_map
        except Exception as e:
            print(f"[Depth Engine] Depth Anything V2 failed: {e}. Generating synthetic linear depth gradient.")
            return cls.generate_synthetic_depth_map(h, w, mask)

    @classmethod
    def process_depth_blur(
        cls,
        pil_image: Image.Image,
        mask_pil: Image.Image,
        blur_mode: str = "Lens Blur / Circular Bokeh (Realistic DSLR)",
        blur_preset: str = "DSLR 85mm",
        blur_strength: float = 45.0,
        edge_feathering: int = 5,
        subject_protection: float = 85.0,
        background_smoothness: float = 30.0
    ) -> dict:
        """
        Executes the complete spatially-varying optical depth blur pipeline.
        Returns a dictionary containing:
          - "result": PIL composite image with depth blur
          - "depth_map": PIL depth map image (for visual developer previews)
        """
        # 1. Convert formats
        cv_img = pil_to_cv(pil_image)
        mask = np.array(mask_pil.convert("L"))
        h, w = cv_img.shape[:2]
        
        # Ensure matching shapes
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
        # 2. Pre-generate or run depth estimation
        depth = cls.run_depth_estimation(cv_img, mask)
        
        # 3. Soft-feather the mask
        alpha = DSLRBlurProcessor.apply_feathering(mask, edge_feathering)
        
        # 4. Calibrate Focus Plane Depth (D_subject) to the isolated subject
        subject_pixels = depth[mask > 127]
        if len(subject_pixels) > 0:
            D_subject = float(np.median(subject_pixels))
        else:
            D_subject = 220.0 # Default foreground depth focus
            
        # 5. Map Preset parameters to Max Blur Strength
        # Presets: Portrait, DSLR 50mm, DSLR 85mm, Studio, Cinematic, Custom
        preset_max_blur = {
            "Portrait": 25.0,
            "DSLR 50mm": 40.0,
            "DSLR 85mm": 55.0,
            "Studio": 20.0,
            "Cinematic": 75.0,
            "Custom": blur_strength
        }
        max_blur = preset_max_blur.get(blur_preset, blur_strength)
        
        # 6. Calculate Spatially-Varying Blur Radius map
        # Dist from focus plane scaled [0.0, 1.0]
        dist_from_focus = np.abs(depth.astype(np.float32) - D_subject) / 255.0
        
        # Blur radius maps from 0 to max_blur (subject has 0 blur because of (1 - alpha))
        radius_map = max_blur * (1.0 - alpha) * dist_from_focus
        
        # 7. Reconstruct Background to prevent colored halos/subject bleeding
        bg_inpainted = DSLRBlurProcessor.inpaint_background(cv_img[:, :, :3], mask)
        
        # 8. Vectorized N-Layer Depth-of-Field Blending
        # Generate N=6 discrete blur radius stops for linear interpolation
        radii = [0, int(0.15 * max_blur), int(0.35 * max_blur), int(0.60 * max_blur), int(0.85 * max_blur), int(max_blur)]
        # Filter duplicates and ensure they are sorted ascending
        radii = sorted(list(set(radii)))
        
        # Pre-compute N blurred background layers
        blurred_layers = []
        for r in radii:
            if r == 0:
                blurred_layers.append(bg_inpainted)
            else:
                # Ensure the radius maps to an odd number internally in apply_blur
                blurred_layers.append(DSLRBlurProcessor.apply_blur(bg_inpainted, blur_mode, r, background_smoothness))
                
        # Perform vectorized linear interpolation across depth intervals
        composite_bg = np.zeros_like(cv_img[:, :, :3], dtype=np.float32)
        
        for k in range(len(radii) - 1):
            r_low = radii[k]
            r_high = radii[k+1]
            
            # Identify pixels whose target blur radius falls inside this interval
            if k == len(radii) - 2:
                interval_mask = (radius_map >= r_low) & (radius_map <= r_high)
            else:
                interval_mask = (radius_map >= r_low) & (radius_map < r_high)
                
            if not np.any(interval_mask):
                continue
                
            # Grab target radii values in this interval
            r_vals = radius_map[interval_mask]
            
            # Linear interpolation weight: w = 1.0 at low_radius, 0.0 at high_radius
            w_interp = (r_high - r_vals) / (r_high - r_low)
            w_interp_3d = np.expand_dims(w_interp, axis=1) # shape (num_pixels, 1)
            
            # Slice pixels from adjacent blurred layers
            low_pixels = blurred_layers[k][interval_mask].astype(np.float32)
            high_pixels = blurred_layers[k+1][interval_mask].astype(np.float32)
            
            # Blend pixels and write back
            blended_pixels = low_pixels * w_interp_3d + high_pixels * (1.0 - w_interp_3d)
            composite_bg[interval_mask] = blended_pixels
            
        # Clip to valid range and cast
        bg_depth_blurred = np.clip(composite_bg, 0, 255).astype(np.uint8)
        
        # 9. Composite the original sharp subject over this depth-blurred background
        result_cv = DSLRBlurProcessor.composite_layers(cv_img[:, :, :3], bg_depth_blurred, alpha, subject_protection)
        
        # 10. Re-apply alpha transparency if present
        if cv_img.shape[2] == 4:
            result_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            result_rgba[:, :, :3] = result_cv
            result_rgba[:, :, 3] = cv_img[:, :, 3]
            output_pil = cv_to_pil(result_rgba)
        else:
            output_pil = cv_to_pil(result_cv)
            
        return {
            "result": output_pil,
            "depth_map": Image.fromarray(depth).convert("L")
        }
