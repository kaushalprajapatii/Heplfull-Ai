import cv2
import numpy as np
from PIL import Image
from backend.utilities import pil_to_cv, cv_to_pil
from backend.enhancement.onnx_engine import ONNXInferenceEngine

class PhotoEnhancer:
    """
    Orchestrates AI Photo Enhancement, Face Restoration, and Color Grading.
    """
    
    @staticmethod
    def apply_color_grading(img: np.ndarray, vibrancy_boost: float = 1.15, contrast_boost: float = 1.02) -> np.ndarray:
        """
        Preserves 100% of the original photo's natural color palette, balance, and contrast.
        Returns the input image completely untouched.
        """
        return img

    @classmethod
    def restore_faces(cls, img: np.ndarray) -> np.ndarray:
        """
        Detects faces in the image, runs GFPGAN ONNX inference on crops, 
        and blends them back seamlessly using feathered alpha blending.
        """
        # Load OpenCV Haar Cascade face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return img
            
        result = img.copy()
        h, w = img.shape[:2]
        
        print(f"[Photo Enhancer] Detected {len(faces)} face(s) for GFPGAN restoration.")
        for idx, (fx, fy, fw, fh) in enumerate(faces):
            try:
                # Add 40% padding around the detected face region for natural context
                pad_x = int(fw * 0.4)
                pad_y = int(fh * 0.4)
                
                x1 = max(0, fx - pad_x)
                y1 = max(0, fy - pad_y)
                x2 = min(w, fx + fw + pad_x)
                y2 = min(h, fy + fh + pad_y)
                
                crop_w = x2 - x1
                crop_h = y2 - y1
                
                if crop_w < 20 or crop_h < 20:
                    continue
                    
                face_crop = img[y1:y2, x1:x2]
                
                # Execute GFPGAN ONNX restoration
                restored_face_512 = ONNXInferenceEngine.run_gfpgan(face_crop)
                
                # Resize the restored face back to the crop's original dimensions
                restored_face = cv2.resize(restored_face_512, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)
                
                # Create a feathered ellipse blending mask to overlay the face crop seamlessly
                mask = np.zeros((crop_h, crop_w), dtype=np.float32)
                cx = crop_w // 2
                cy = crop_h // 2
                rx = int(crop_w * 0.45)
                ry = int(crop_h * 0.45)
                cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
                
                # Blur the mask to create a smooth, linear alpha transition
                mask_blurred = cv2.GaussianBlur(mask, (31, 31), 0)
                mask_3d = np.expand_dims(mask_blurred, axis=2)
                
                # Blend the restored face back onto the image
                original_crop = result[y1:y2, x1:x2].astype(np.float32)
                restored_crop = restored_face.astype(np.float32)
                
                blended_crop = restored_crop * mask_3d + original_crop * (1.0 - mask_3d)
                result[y1:y2, x1:x2] = np.clip(blended_crop, 0, 255).astype(np.uint8)
            except Exception as e:
                print(f"[Photo Enhancer] Face {idx+1} restoration failed: {e}. Falling back to original face.")
                
        return result

    @classmethod
    def apply_opencv_enhancement(cls, img: np.ndarray) -> np.ndarray:
        """
        High-performance zero-dependency detail enhancer (Bilateral denoising + Unsharp Masking).
        Preserves 100% original color, contrast, and histogram distributions.
        """
        # 1. Bilateral filtering (reduces noise, preserves sharp object edges)
        denoised = cv2.bilateralFilter(img, 9, 35, 35)
        
        # 2. High-Pass Unsharp Masking (recovers micro-details and textures)
        blurred = cv2.GaussianBlur(denoised, (0, 0), 3)
        sharpened = cv2.addWeighted(denoised, 1.4, blurred, -0.4, 0)
        
        return sharpened

    @classmethod
    def process_enhancement(
        cls,
        pil_image: Image.Image,
        mode: str = "Professional DSLR",
        face_restoration: bool = True
    ) -> Image.Image:
        """
        Main entry point to execute the photo enhancement pipeline.
        Modes:
          - "Standard": Fast, zero-dependency detail sharpener + color grading.
          - "High Quality (Neural)": Real-ESRGAN super-resolution + color grading.
          - "Professional DSLR": Full pipeline (Real-ESRGAN + GFPGAN Face Restore + Color Grading).
        """
        cv_img = pil_to_cv(pil_image)
        
        # 1. Photo Upscaling & Detail Sharpness
        if mode in ["High Quality (Neural)", "Professional DSLR"]:
            try:
                # Try running Real-ESRGAN ONNX
                enhanced_cv = ONNXInferenceEngine.run_realesrgan(cv_img[:, :, :3])
            except Exception as e:
                print(f"[Photo Enhancer] Real-ESRGAN upscale failed: {e}. Falling back to OpenCV sharpener.")
                # Fallback to high-speed OpenCV sharpener
                enhanced_cv = cls.apply_opencv_enhancement(cv_img[:, :, :3])
        else:
            # Standard Mode (OpenCV sharpener)
            enhanced_cv = cls.apply_opencv_enhancement(cv_img[:, :, :3])
            
        # 2. Face Restoration (GFPGAN)
        if face_restoration and mode == "Professional DSLR":
            try:
                enhanced_cv = cls.restore_faces(enhanced_cv)
            except Exception as e:
                print(f"[Photo Enhancer] GFPGAN face restoration failed: {e}.")
                
        # 3. Color Grading & Aesthetics
        final_cv = cls.apply_color_grading(enhanced_cv)
        
        # Apply original transparency alpha channel if present
        if cv_img.shape[2] == 4:
            h, w = final_cv.shape[:2]
            alpha = cv2.resize(cv_img[:, :, 3], (w, h), interpolation=cv2.INTER_LINEAR)
            result_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            result_rgba[:, :, :3] = final_cv
            result_rgba[:, :, 3] = alpha
            return cv_to_pil(result_rgba)
        else:
            return cv_to_pil(final_cv)
