import cv2
import numpy as np
from PIL import Image
from backend.utilities import pil_to_cv, cv_to_pil

class DSLRBlurProcessor:
    """
    Orchestrator class to execute the DSLR Background Blur image processing pipeline.
    """
    
    @staticmethod
    def apply_feathering(mask: np.ndarray, radius: int) -> np.ndarray:
        """
        Feathers the binary mask to create a soft, anti-aliased edge transition.
        Returns a float32 mask scaled between 0.0 and 1.0.
        """
        if radius <= 0:
            return mask.astype(np.float32) / 255.0
            
        # Ensure kernel size is odd
        k_size = radius * 2 + 1
        feathered = cv2.GaussianBlur(mask, (k_size, k_size), 0)
        return feathered.astype(np.float32) / 255.0

    @staticmethod
    def inpaint_background(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaints/erases the foreground subject out of the background.
        Uses a highly optimized downscaled inpainting approach to prevent color bleeding
        and edge-halos when the background gets blurred.
        """
        h, w = img.shape[:2]
        
        # 1. Dilate the mask by 15px to fully cover edge transition and anti-aliasing zones
        kernel_size = 15
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        
        # 2. Downscale the image and mask to 25% size for lightning-fast inpainting
        scale = 0.25
        down_w = int(w * scale)
        down_h = int(h * scale)
        
        img_small = cv2.resize(img, (down_w, down_h), interpolation=cv2.INTER_AREA)
        mask_small = cv2.resize(dilated_mask, (down_w, down_h), interpolation=cv2.INTER_NEAREST)
        
        # 3. Perform Fast Telea inpainting on the downscaled image
        inpainted_small = cv2.inpaint(img_small, mask_small, 5, cv2.INPAINT_TELEA)
        
        # 4. Upscale back to the original image dimensions
        inpainted = cv2.resize(inpainted_small, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # 5. Composite back the real background pixels (keeping inpainted pixels only under dilated mask)
        bg_only = img.copy()
        mask_indices = dilated_mask > 0
        bg_only[mask_indices] = inpainted[mask_indices]
        
        return bg_only

    @staticmethod
    def apply_blur(img: np.ndarray, mode: str, strength: float, smoothness: float) -> np.ndarray:
        """
        Applies a natural, aesthetically pleasing blur to the background.
        Supports:
          - "Gaussian Blur (Soft & Smooth)"
          - "Lens Blur / Circular Bokeh (Realistic DSLR)"
        """
        # Map strength (1 - 100) to actual kernel/radius dimensions
        # For Gaussian: map to odd numbers from 3 to 101
        g_strength = int(strength / 100.0 * 50.0) * 2 + 1
        g_strength = max(3, g_strength)
        
        # For Circular Bokeh: map circular kernel diameter from 3 to 61
        l_diameter = int(strength / 100.0 * 30.0) * 2 + 1
        l_diameter = max(3, l_diameter)
        
        if mode == "Lens Blur / Circular Bokeh (Realistic DSLR)":
            # Create a flat circular convolution kernel representing lens aperture
            kernel = np.zeros((l_diameter, l_diameter), dtype=np.float32)
            cv2.circle(kernel, (l_diameter // 2, l_diameter // 2), l_diameter // 2, 1, -1)
            
            # Normalize the kernel
            kernel_sum = np.sum(kernel)
            if kernel_sum > 0:
                kernel /= kernel_sum
            else:
                kernel[l_diameter // 2, l_diameter // 2] = 1.0
                
            # Convolve background to form circular bokeh discs
            blurred = cv2.filter2D(img, -1, kernel)
        else:
            # Gaussian Blur
            blurred = cv2.GaussianBlur(img, (g_strength, g_strength), 0)
            
        # Bilateral filter post-smoothing for a creamy, noise-free studio look
        if smoothness > 0:
            d = int(smoothness / 100.0 * 15)
            d = max(3, d | 1)  # must be odd
            sigma_color = smoothness / 100.0 * 150.0
            sigma_space = smoothness / 100.0 * 150.0
            blurred = cv2.bilateralFilter(blurred, d, sigma_color, sigma_space)
            
        return blurred

    @staticmethod
    def composite_layers(
        fg_img: np.ndarray, 
        bg_img: np.ndarray, 
        alpha: np.ndarray, 
        subject_protection: float
    ) -> np.ndarray:
        """
        Composites the sharp foreground subject over the blurred background.
        alpha: float32 grayscale feathered mask in range [0, 1.0]. Shape is (H, W).
        subject_protection: float (0 - 100) -> Protects original fine details.
        """
        # Expand alpha to 3 channels for RGB broadcasting
        alpha_3d = np.expand_dims(alpha, axis=2).copy()
        
        # Smart Subject Protection:
        # Protects the solid core of the subject (alpha > 0.7), but NEVER clamps the feathered edge
        # (alpha <= 0.7). This ensures boundaries fade smoothly to 0.0, completely eliminating
        # the ugly "cut-out sticker" halo of unblurred background/grass around the subject.
        if subject_protection > 0:
            protection_factor = subject_protection / 100.0
            # Only apply protection clamping inside the subject core
            mask_core = alpha > 0.70
            alpha_3d[mask_core] = np.maximum(alpha_3d[mask_core], protection_factor)
            
        # Alpha blend: out = fg * alpha + bg * (1 - alpha)
        composited = fg_img.astype(np.float32) * alpha_3d + bg_img.astype(np.float32) * (1.0 - alpha_3d)
        return np.clip(composited, 0, 255).astype(np.uint8)

    @classmethod
    def process_dslr_blur(
        cls,
        pil_image: Image.Image,
        mask_pil: Image.Image,
        blur_mode: str = "Lens Blur / Circular Bokeh (Realistic DSLR)",
        blur_strength: float = 30.0,
        edge_feathering: int = 5,
        subject_protection: float = 80.0,
        background_smoothness: float = 30.0
    ) -> Image.Image:
        """
        Main entry point to execute the DSLR Background Blur pipeline.
        """
        # 1. Convert to CV BGR/BGRA arrays
        cv_img = pil_to_cv(pil_image)
        mask = np.array(mask_pil.convert("L"))
        
        # Ensure matching shapes
        h, w = cv_img.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
        # 2. Feather the mask to create anti-aliased subject edges
        alpha = cls.apply_feathering(mask, edge_feathering)
        
        # 3. Inpaint the background to erase the subject and prevent colored edge halos/bleeding
        bg_inpainted = cls.inpaint_background(cv_img[:, :, :3], mask)
        
        # 4. Apply Gaussian or circular lens bokeh blur to the background
        bg_blurred = cls.apply_blur(bg_inpainted, blur_mode, blur_strength, background_smoothness)
        
        # 5. Composite original sharp subject over the blurred background using feathered alpha
        result_cv = cls.composite_layers(cv_img[:, :, :3], bg_blurred, alpha, subject_protection)
        
        # 6. Re-apply alpha channel if original image was RGBA
        if cv_img.shape[2] == 4:
            result_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            result_rgba[:, :, :3] = result_cv
            result_rgba[:, :, 3] = cv_img[:, :, 3]
            return cv_to_pil(result_rgba)
        else:
            return cv_to_pil(result_cv)
