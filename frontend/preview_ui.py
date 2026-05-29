import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import cv2
import io
from backend.utilities import pil_to_cv, cv_to_pil, create_checkerboard
from backend.bg_remover.shadow_generator import blend_rgba

def generate_gradient_background(w: int, h: int, style: str) -> np.ndarray:
    """Generate a custom studio gradient background (RGBA)."""
    bg = np.zeros((h, w, 4), dtype=np.uint8)
    bg[:, :, 3] = 255  # fully opaque
    
    if style == "Midnight Glow":
        # Radial gradient from dark indigo (#1A1B35) to black (#0B0C10)
        cx, cy = w // 2, h // 2
        max_dist = np.sqrt(cx**2 + cy**2)
        y, x = np.indices((h, w))
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        factor = np.clip(dist / max_dist, 0.0, 1.0)
        
        # Color 1: 26, 27, 53  | Color 2: 11, 12, 16
        for c in range(3):
            c1, c2 = [26, 27, 53][c], [11, 12, 16][c]
            bg[:, :, c] = (c1 * (1.0 - factor) + c2 * factor).astype(np.uint8)
            
    elif style == "Sunset Studio":
        # Linear vertical gradient from soft orange (#FF7E5F) to deep purple (#FEB47B)
        y, _ = np.indices((h, w))
        factor = (y / h).astype(float)
        
        # Color 1: 255, 126, 95 | Color 2: 254, 180, 123
        for c in range(3):
            c1, c2 = [255, 126, 95][c], [254, 180, 123][c]
            bg[:, :, c] = (c1 * (1.0 - factor) + c2 * factor).astype(np.uint8)
            
    elif style == "Clean Studio":
        # Vertical gradient from light gray (#F5F7FA) to slate gray (#B3C0CD)
        y, _ = np.indices((h, w))
        factor = (y / h).astype(float)
        
        # Color 1: 245, 247, 250 | Color 2: 179, 192, 205
        for c in range(3):
            c1, c2 = [245, 247, 250][c], [179, 192, 205][c]
            bg[:, :, c] = (c1 * (1.0 - factor) + c2 * factor).astype(np.uint8)
            
    elif style == "Neon Cyber":
        # Diagonal gradient from hot magenta (#FF007F) to cyber cyan (#00F0FF)
        y, x = np.indices((h, w))
        # Normalized coordinates summing for diagonal
        factor = np.clip((x / w + y / h) / 2.0, 0.0, 1.0)
        
        # Color 1: 255, 0, 127 | Color 2: 0, 240, 255
        for c in range(3):
            c1, c2 = [255, 0, 127][c], [0, 240, 255][c]
            bg[:, :, c] = (c1 * (1.0 - factor) + c2 * factor).astype(np.uint8)
            
    return bg

def composite_cutout_on_bg(cutout_pil: Image.Image, bg_type: str, custom_color: str = "#FFFFFF", custom_bg_file = None) -> Image.Image:
    """Composite the transparent cutout over a selected background style."""
    w, h = cutout_pil.size
    cutout_cv = pil_to_cv(cutout_pil) # RGBA BGR/BGRA
    
    # Ensure BG is shape (H, W, 4)
    bg_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    bg_rgba[:, :, 3] = 255  # solid background by default
    
    if bg_type == "Transparent (Checkerboard)":
        bg_rgba = create_checkerboard(w, h, square_size=16)
        
    elif bg_type == "Solid White":
        bg_rgba[:, :, :3] = 255
        
    elif bg_type == "Solid Black":
        bg_rgba[:, :, :3] = 0
        
    elif bg_type == "Custom Solid Color":
        # Convert hex custom_color "#RRGGBB" to BGR
        hex_color = custom_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        bg_rgba[:, :, 0] = b
        bg_rgba[:, :, 1] = g
        bg_rgba[:, :, 2] = r
        
    elif bg_type in ["Midnight Glow", "Sunset Studio", "Clean Studio", "Neon Cyber"]:
        bg_rgba = generate_gradient_background(w, h, bg_type)
        
    elif bg_type == "Custom Upload Image" and custom_bg_file is not None:
        try:
            # Load and resize custom background to match the cutout dimensions
            custom_bg_pil = Image.open(custom_bg_file).convert("RGBA")
            custom_bg_resized = custom_bg_pil.resize((w, h), Image.Resampling.LANCZOS)
            bg_rgba = pil_to_cv(custom_bg_resized)
        except Exception as e:
            # Fallback to white if error
            bg_rgba[:, :, :3] = 255
            st.error(f"Error loading custom background: {str(e)}")
            
    # Perform alpha blending
    blended_cv = blend_rgba(top=cutout_cv, bottom=bg_rgba)
    return cv_to_pil(blended_cv)

def render_preview_ui(
    original_pil: Image.Image,
    processed_pil: Image.Image,
    bounding_box: tuple = None,
    show_bounding_box: bool = False
):
    """
    Renders the beautiful side-by-side image previews with interactive controls.
    """
    st.markdown(
        """
        <div style="margin-top: 25px; margin-bottom: 15px;">
            <h3 style="color: #00F0FF; font-weight: 600; letter-spacing: 0.5px; margin: 0 0 10px 0;">
                🖼️ VISUAL COMPARISON
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 1. Bounding Box visual overlay helper
    display_original = original_pil.copy()
    if show_bounding_box and bounding_box is not None:
        x, y, w, h = bounding_box
        draw = ImageDraw.Draw(display_original)
        # Draw elegant bounding box with thick line and glowing color (#00F0FF)
        draw.rectangle([x, y, x + w, y + h], outline="#00F0FF", width=3)
        # Optional: Add small label
        draw.text((x + 5, y + 5), "GrabCut Bounding Box", fill="#00F0FF")
        
    # 2. Render background selector in preview options
    st.markdown('<div class="glass-card" style="padding: 15px; margin-bottom: 20px;">', unsafe_allow_html=True)
    col_bg, col_color, col_custom_upload = st.columns([2, 1, 2])
    
    with col_bg:
        bg_preview_type = st.selectbox(
            "Background Preview Mode",
            [
                "Transparent (Checkerboard)",
                "Solid White",
                "Solid Black",
                "Custom Solid Color",
                "Clean Studio",
                "Midnight Glow",
                "Sunset Studio",
                "Neon Cyber",
                "Custom Upload Image"
            ],
            index=0,
            key="bg_preview_select"
        )
        
    with col_color:
        custom_color = "#FFFFFF"
        if bg_preview_type == "Custom Solid Color":
            custom_color = st.color_picker("Pick Color", "#6C63FF", key="custom_color_picker")
        else:
            st.write("") # placeholder
            
    with col_custom_upload:
        custom_bg_file = None
        if bg_preview_type == "Custom Upload Image":
            custom_bg_file = st.file_uploader("Upload Backdrop Image", type=["jpg", "jpeg", "png", "webp"], key="custom_bg_uploader")
        else:
            st.write("") # placeholder
            
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 3. Composite output cutout on selected background
    composited_processed = composite_cutout_on_bg(
        processed_pil, 
        bg_preview_type, 
        custom_color=custom_color, 
        custom_bg_file=custom_bg_file
    )
    
    # 4. Display the columns
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown('<p class="preview-label">ORIGINAL SOURCE IMAGE</p>', unsafe_allow_html=True)
        st.image(display_original, use_container_width=True)
        
    with col_right:
        st.markdown('<p class="preview-label">EXTRACTED SUBJECT PREVIEW</p>', unsafe_allow_html=True)
        st.image(composited_processed, use_container_width=True)
