import streamlit as st
from PIL import Image
import io

def convert_pil_to_bytes(pil_img: Image.Image) -> bytes:
    """Helper to convert a PIL Image into a downloadable byte stream (PNG)."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def render_download_ui(transparent_pil: Image.Image, shadow_pil: Image.Image, filename: str = "cutout"):
    """
    Renders styled download buttons for the transparent cutout and shadow-composite images.
    """
    st.markdown(
        """
        <div style="margin-top: 30px; margin-bottom: 15px;">
            <h3 style="color: #00F0FF; font-weight: 600; letter-spacing: 0.5px; margin: 0 0 10px 0;">
                💾 EXPORT RESULT
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Pre-render images to bytes
    with st.spinner("Preparing high-quality files for export..."):
        transparent_bytes = convert_pil_to_bytes(transparent_pil)
        
        # Check if shadow is enabled (i.e. different from transparent_pil)
        # In our case, we can check if they are separate or just present both
        shadow_bytes = convert_pil_to_bytes(shadow_pil)
        
    st.markdown('<div class="glass-card" style="padding: 20px;">', unsafe_allow_html=True)
    
    col_dl1, col_dl2 = st.columns(2)
    
    # Strip extension from filename for the download prefix
    base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
    
    with col_dl1:
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 10px;">
                <span style="font-size: 1.8rem;">✨</span>
                <p style="font-weight: 600; margin: 5px 0 0 0; color: #FFF;">Transparent PNG</p>
                <p style="font-size: 0.75rem; color: #8A99AD; margin: 2px 0 10px 0;">Best for composition, overlays, and design work.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.download_button(
            label="Download Transparent Cutout",
            data=transparent_bytes,
            file_name=f"{base_name}_cutout.png",
            mime="image/png",
            key="dl_transparent",
            use_container_width=True
        )
        
    with col_dl2:
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 10px;">
                <span style="font-size: 1.8rem;">🌗</span>
                <p style="font-weight: 600; margin: 5px 0 0 0; color: #FFF;">Realistic Shadow Composite</p>
                <p style="font-size: 0.75rem; color: #8A99AD; margin: 2px 0 10px 0;">Includes refined transparent subject + drop shadow.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.download_button(
            label="Download Shadow Composite",
            data=shadow_bytes,
            file_name=f"{base_name}_shadow.png",
            mime="image/png",
            key="dl_shadow",
            use_container_width=True
        )
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Optional tip info
    st.markdown(
        """
        <p style="text-align: center; color: #8A99AD; font-size: 0.75rem; margin-top: 15px;">
            💡 <b>Pro-Tip:</b> All images are exported as lossless, high-fidelity <b>RGBA PNG</b> format to preserve the soft feathered edges and custom opacity shadows.
        </p>
        """,
        unsafe_allow_html=True
    )
