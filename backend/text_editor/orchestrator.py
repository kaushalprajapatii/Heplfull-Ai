import numpy as np
from PIL import Image
from backend.utilities import pil_to_cv, cv_to_pil
from backend.text_editor.ocr_engine import detect_text_in_image
from backend.text_editor.color_detector import detect_ink_and_paper_colors
from backend.text_editor.text_replacer import replace_text_in_image

class TextEditorOrchestrator:
    @staticmethod
    def run_ocr(pil_image: Image.Image) -> list:
        """
        Runs OCR on the given PIL image to detect all text fields.
        
        Returns:
        List of dicts: [{"text": str, "bbox": (x, y, w, h), "confidence": float}]
        """
        if pil_image is None:
            return []
            
        img_bgr = pil_to_cv(pil_image)
        return detect_text_in_image(img_bgr)
        
    @staticmethod
    def apply_replacements(
        pil_image: Image.Image,
        replacements: list,
        font_family: str = "Sans-Serif",
        size_multiplier: float = 0.85
    ) -> Image.Image:
        """
        Applies a list of text replacements to the image.
        
        Parameters:
        - pil_image: The original PIL Image.
        - replacements: List of dicts, each with:
             - "bbox": (x, y, w, h)
             - "replacement_text": str
        - font_family: Font family style to use ("Sans-Serif", "Serif", "Monospace").
        - size_multiplier: Text size scale multiplier.
        
        Returns:
        The edited PIL Image.
        """
        if pil_image is None or not replacements:
            return pil_image
            
        # Make a copy of the image and convert to OpenCV BGR
        img_bgr = pil_to_cv(pil_image).copy()
        
        for rep in replacements:
            bbox = rep.get("bbox")
            new_text = rep.get("replacement_text")
            
            if bbox is None or new_text is None:
                continue
                
            # 1. Detect ink and paper colors in the bounding box
            ink_color, paper_color = detect_ink_and_paper_colors(img_bgr, bbox)
            
            # 2. Erase, inpaint, and synthesize replacement text
            img_bgr = replace_text_in_image(
                img_bgr=img_bgr,
                bbox=bbox,
                replacement_text=new_text,
                ink_color_bgr=ink_color,
                paper_color_bgr=paper_color,
                font_family=font_family,
                size_multiplier=size_multiplier
            )
            
        # Convert back to PIL Image
        return cv_to_pil(img_bgr)
