import os
import cv2
import numpy as np
import urllib.request
from PIL import Image

class ONNXInferenceEngine:
    """
    Handles downloading, caching, and inference of ONNX models for:
    - Depth Anything V2 (Depth Estimation)
    - GFPGAN v1.4 (Face Restoration)
    - Real-ESRGAN x2 (Super Resolution & Artifact Removal)
    """
    
    MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    
    MODEL_URLS = {
        "depth_anything": "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx", # 97 MB
        "gfpgan": "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx",                       # 140 MB
        "realesrgan": "https://huggingface.co/tidus2102/Real-ESRGAN/resolve/main/Real-ESRGAN_x2plus.onnx"                # 67 MB
    }
    
    _failed_models = set()
    
    @classmethod
    def get_model_path(cls, model_key: str) -> str:
        """Gets local path to the model, downloading it if not present."""
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
        filename = f"{model_key}.onnx"
        dest_path = os.path.join(cls.MODELS_DIR, filename)
        
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10 * 1024 * 1024:
            return dest_path
            
        url = cls.MODEL_URLS[model_key]
        temp_path = dest_path + ".tmp"
        
        print(f"[ONNX Engine] Downloading {model_key} model weights (~{cls._get_size_str(model_key)})...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(temp_path, "wb") as out_file:
                block_size = 1024 * 1024 # 1 MB chunks
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    out_file.write(buffer)
            os.replace(temp_path, dest_path)
            print(f"[ONNX Engine] Successfully downloaded and cached {model_key} model.")
            return dest_path
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"[ONNX Engine] Failed to download {model_key}: {e}")
            raise e

    @staticmethod
    def _get_size_str(model_key: str) -> str:
        sizes = {"depth_anything": "97MB", "gfpgan": "140MB", "realesrgan": "67MB"}
        return sizes.get(model_key, "Unknown")

    @classmethod
    def get_session(cls, model_key: str):
        """Initializes and returns an ONNX Runtime inference session."""
        if model_key in cls._failed_models:
            print(f"[ONNX Engine] Skipping {model_key} session loading due to previous failures in this run.")
            return None
        try:
            import onnxruntime as ort
            model_path = cls.get_model_path(model_key)
            # Use CPU execution provider with optimized thread configurations
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
            return session
        except Exception as e:
            print(f"[ONNX Engine] Error loading {model_key} session: {e}")
            cls._failed_models.add(model_key)
            return None

    @classmethod
    def run_depth_anything(cls, cv_img: np.ndarray) -> np.ndarray:
        """
        Executes Depth Anything V2 Small ONNX model to extract a relative depth map.
        Input: cv_img (BGR numpy array). Output: Grayscale depth map normalized to [0, 255].
        """
        session = cls.get_session("depth_anything")
        if session is None:
            raise RuntimeError("Depth Anything V2 ONNX session is not available.")
            
        h, w = cv_img.shape[:2]
        
        # 1. Preprocess: Resize to 518x518 (multiple of 14, model input standard)
        input_size = 518
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (input_size, input_size), interpolation=cv2.INTER_AREA)
        
        # Normalize with ImageNet stats
        img_float = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_float - mean) / std
        
        # Transpose to CHW (3, 518, 518) and add batch dim
        img_tensor = np.transpose(img_norm, (2, 0, 1))
        img_tensor = np.expand_dims(img_tensor, axis=0)
        
        # 2. Run Inference
        inputs = {session.get_inputs()[0].name: img_tensor}
        depth_out = session.run(None, inputs)[0]
        
        # Depth output shape is (1, 1, 518, 518) or (1, 518, 518)
        depth_map = np.squeeze(depth_out)
        
        # 3. Postprocess: Min-Max normalize to [0, 255]
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max > d_min:
            depth_map = (depth_map - d_min) / (d_max - d_min) * 255.0
        else:
            depth_map = np.zeros_like(depth_map)
            
        depth_map = depth_map.astype(np.uint8)
        
        # Resize back to original dimensions using bilinear interpolation
        depth_original = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
        return depth_original

    @classmethod
    def run_gfpgan(cls, face_bgr: np.ndarray) -> np.ndarray:
        """
        Executes GFPGAN v1.4 ONNX model on a cropped facial BGR image.
        Input: face_bgr (BGR array, expected 512x512). Output: Restored BGR face.
        """
        session = cls.get_session("gfpgan")
        if session is None:
            raise RuntimeError("GFPGAN ONNX session is not available.")
            
        # 1. Preprocess: Resize to 512x512
        face_resized = cv2.resize(face_bgr, (512, 512), interpolation=cv2.INTER_AREA)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1.0, 1.0]
        face_float = face_rgb.astype(np.float32)
        face_norm = (face_float - 127.5) / 127.5
        
        # Transpose to CHW and expand batch
        face_tensor = np.transpose(face_norm, (2, 0, 1))
        face_tensor = np.expand_dims(face_tensor, axis=0)
        
        # 2. Run Inference
        inputs = {session.get_inputs()[0].name: face_tensor}
        restored_out = session.run(None, inputs)[0]
        
        # 3. Postprocess: Scale back to [0, 255]
        restored_face = np.squeeze(restored_out)
        restored_face = np.transpose(restored_face, (1, 2, 0)) # CHW -> HWC
        restored_face = np.clip((restored_face + 1.0) * 127.5, 0, 255).astype(np.uint8)
        
        # Convert RGB back to BGR
        restored_bgr = cv2.cvtColor(restored_face, cv2.COLOR_RGB2BGR)
        return restored_bgr

    @classmethod
    def run_realesrgan(cls, cv_img: np.ndarray) -> np.ndarray:
        """
        Executes Real-ESRGAN x2 ONNX model to upscale and remove compression artifacts.
        Input: cv_img (BGR). Output: 2x Super-Resolved BGR image.
        """
        session = cls.get_session("realesrgan")
        if session is None:
            raise RuntimeError("Real-ESRGAN ONNX session is not available.")
            
        h, w = cv_img.shape[:2]
        
        # For CPU safety, if the input image is already large, we scale it down before upscaling
        # to prevent memory overflows and extremely slow processing.
        max_input_dim = 700
        if max(h, w) > max_input_dim:
            if h > w:
                new_h = max_input_dim
                new_w = int(w * (max_input_dim / h))
            else:
                new_w = max_input_dim
                new_h = int(h * (max_input_dim / w))
            processing_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            processing_img = cv_img.copy()
            
        ph, pw = processing_img.shape[:2]
        
        # 1. Preprocess: Convert to RGB and normalize to [0.0, 1.0]
        img_rgb = cv2.cvtColor(processing_img, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW and add batch dimension
        img_tensor = np.transpose(img_float, (2, 0, 1))
        img_tensor = np.expand_dims(img_tensor, axis=0)
        
        # 2. Run Inference
        inputs = {session.get_inputs()[0].name: img_tensor}
        upscaled_out = session.run(None, inputs)[0]
        
        # 3. Postprocess
        upscaled_img = np.squeeze(upscaled_out)
        upscaled_img = np.transpose(upscaled_img, (1, 2, 0)) # CHW -> HWC
        upscaled_img = np.clip(upscaled_img * 255.0, 0, 255).astype(np.uint8)
        
        # Convert RGB to BGR
        upscaled_bgr = cv2.cvtColor(upscaled_img, cv2.COLOR_RGB2BGR)
        return upscaled_bgr
