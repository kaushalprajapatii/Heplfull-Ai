import streamlit as st
from PIL import Image
import io

def render_upload_ui() -> Image.Image:
    """
    Renders a premium, glassmorphism-styled drag-and-drop file uploader.
    Validates files and returns a PIL Image if successful.
    """
    # Custom styled card for the upload area
    st.markdown(
        """
        <div class="glass-card upload-container">
            <h3 style="text-align: center; color: #00F0FF; margin-top: 0; font-weight: 600; letter-spacing: 0.5px;">
                ⚡ UPLOAD SOURCE IMAGE
            </h3>
            <p style="text-align: center; color: #8A99AD; font-size: 0.9rem; margin-bottom: 20px;">
                Drag and drop your image or browse files. Supports high-resolution JPG, PNG, and WEBP.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Upload widget
    uploaded_file = st.file_uploader(
        "Choose an image file...",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        key="image_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Read file bytes
            file_bytes = uploaded_file.read()
            # Convert to PIL Image
            pil_img = Image.open(io.BytesIO(file_bytes))
            
            # Automatically normalize EXIF camera orientation tags immediately on upload
            from PIL import ImageOps
            pil_img = ImageOps.exif_transpose(pil_img)
            
            # Display success message and metadata in an elegant micro-card
            st.markdown(
                f"""
                <div class="success-badge">
                    <span>✓</span> Image loaded successfully: <b>{uploaded_file.name}</b> ({pil_img.width}x{pil_img.height}px | {pil_img.mode})
                </div>
                """,
                unsafe_allow_html=True
            )
            return pil_img
        except Exception as e:
            st.error(f"Error loading image: {str(e)}. Please try another file.")
            return None
            
    return None
