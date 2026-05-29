import streamlit as st
from PIL import Image
import io
import numpy as np
from backend.bg_remover.image_processor import ImageProcessor
from backend.dslr_blur.depth_blur import DepthBlurEngine
from backend.enhancement.photo_enhancer import PhotoEnhancer
from frontend.upload_ui import render_upload_ui

def load_demo_dslr_image():
    """Create a high-quality synthetic demo image representing a studio portrait for DSLR testing."""
    w, h = 600, 600
    img = Image.new("RGB", (w, h), "#E6ECEF") # soft grey studio background
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Draw some complex patterned background details to highlight the depth-blur falloff
    for x in range(30, w, 60):
        # Draw background window panes / grids
        draw.line([x, 0, x, h], fill="#CBD5E1", width=3)
        draw.line([0, x, w, x], fill="#CBD5E1", width=3)
        
    # Draw a mock horizon/floor divider (depth perspective)
    draw.line([0, 420, w, 420], fill="#94A3B8", width=4)
    
    # Draw the main subject (a mock portrait outline representing a person)
    # Head
    draw.ellipse([220, 100, 380, 260], fill="#FDBA74", outline="#F97316", width=3) # head skin-tone
    # Hair
    draw.chord([210, 80, 390, 220], 180, 360, fill="#1E293B") # dark hair block
    # Eyes
    draw.ellipse([260, 160, 280, 180], fill="#0284C7")
    draw.ellipse([320, 160, 340, 180], fill="#0284C7")
    # Smile
    draw.arc([270, 180, 330, 220], 0, 180, fill="#EF4444", width=3)
    # Torso/Clothing
    draw.ellipse([150, 260, 450, 560], fill="#3B82F6", outline="#1D4ED8", width=3) # blue jacket
    
    st.session_state.dslr_original_image = img
    st.session_state.dslr_filename = "demo_studio_portrait.png"
    st.session_state.dslr_stage_cache = None
    st.session_state.dslr_preview_results = None
    st.session_state.dslr_full_res_results = None

def render_dslr_blur_ui():
    """Renders the comprehensive, production-grade AI DSLR Blur and Photo Enhancement workspace."""
    st.markdown(
        """
        <div style="margin-top: 10px; margin-bottom: 20px;">
            <h2 style="color: #7000FF; font-weight: 700; letter-spacing: -0.5px; margin: 0;">📸 Professional AI DSLR Blur & Enhancement</h2>
            <p style="color: #8A99AD; font-size: 0.95rem; margin: 2px 0 0 0;">Transform ordinary mobile snapshots into magazine-cover DSLR portraits. Enhances skin textures, estimates physical depth layers, and simulates optical wide-aperture lenses.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # ----------------- SESSION STATE INITIALIZATION -----------------
    if "dslr_original_image" not in st.session_state:
        st.session_state.dslr_original_image = None
    if "dslr_filename" not in st.session_state:
        st.session_state.dslr_filename = "portrait"
    if "dslr_stage_cache" not in st.session_state:
        st.session_state.dslr_stage_cache = None
    if "dslr_preview_results" not in st.session_state:
        st.session_state.dslr_preview_results = None
    if "dslr_full_res_results" not in st.session_state:
        st.session_state.dslr_full_res_results = None
        
    # ----------------- SIDEBAR CONTROLS -----------------
    st.sidebar.markdown("### ⚙️ PHOTO STUDIO WORKSHOP")
    
    # 1. Image Enhancement Mode
    enhancement_mode = st.sidebar.selectbox(
        "AI Enhancement Mode",
        ["Professional DSLR", "High Quality (Neural)", "Standard"],
        index=0,
        help="Pro DSLR runs Real-ESRGAN super-resolution + GFPGAN face restoration + CLAHE color grading. High Quality utilizes neural upscale. Standard is high-speed OpenCV."
    )
    
    face_restoration = st.sidebar.checkbox(
        "Face Enhancement (GFPGAN)",
        value=True,
        disabled=(enhancement_mode == "Standard"),
        help="Natively restores eyes, teeth, skin textures, and dynamic facial range without creating artificial overprocessing."
    )
    
    # 2. Subject Extraction Engine
    subject_mode = st.sidebar.selectbox(
        "Subject Extraction Mode",
        ["AI BiRefNet (SOTA General)", "AI U²-Net (Legacy Neural)", "General Subject (GrabCut)"],
        index=0,
        help="AI BiRefNet provides state-of-the-art salient detection capturing fine hairs. GrabCut uses automated bounding boxes."
    )
    
    st.sidebar.markdown("#### 🌗 DSLR Aperture Blurring")
    
    # 3. Blur Simulation Engine
    blur_mode = st.sidebar.selectbox(
        "Optical Lens Simulation",
        ["Lens Blur / Circular Bokeh (Realistic DSLR)", "Gaussian Blur (Soft & Smooth)"],
        index=0,
        help="Lens Blur convolves the background to expand specular highlights into circular bokeh diaphragms. Gaussian is standard smooth blur."
    )
    
    # 4. Blur Preset Dropdown
    blur_preset = st.sidebar.selectbox(
        "DSLR Blur Preset",
        ["Portrait", "DSLR 50mm", "DSLR 85mm", "Studio", "Cinematic", "Custom Settings"],
        index=2,
        help="Select a standard DSLR lens preset to configure max aperture blur automatically, or choose Custom Settings."
    )
    
    # Render custom blur strength slider if Custom Settings selected
    is_custom = (blur_preset == "Custom Settings")
    blur_strength = st.sidebar.slider(
        "Max Blur Aperture Strength",
        min_value=1,
        max_value=100,
        value=45,
        disabled=not is_custom,
        help="Enabled under Custom Settings. Adjusts the maximum lens circle of confusion diameter."
    )
    
    edge_feathering = st.sidebar.slider(
        "Edge Feathering Radius",
        min_value=0,
        max_value=30,
        value=5,
        help="Softens the boundaries of the subject to prevent an artificial sharp cutout outline."
    )
    
    subject_protection = st.sidebar.slider(
        "Subject Protection Strength",
        min_value=0,
        max_value=100,
        value=85,
        help="Controls the sharpness threshold inside the subject mask to keep high-frequency details perfectly sharp."
    )
    
    background_smoothness = st.sidebar.slider(
        "Background Smoothness",
        min_value=0,
        max_value=100,
        value=30,
        help="Applies a bilateral filter to suppress sensor noise in the background, making it ultra-creamy."
    )
    
    show_depth_map = st.sidebar.checkbox(
        "Preview Relative Depth Map",
        value=False,
        help="Displays the relative depth map estimated by Depth Anything V2."
    )
    
    # 5. Performance Speed config
    resolution_mode = st.sidebar.radio(
        "Processing Resolution",
        ["High-Speed Preview (Max 800px)", "Full Resolution (High Quality)"],
        key="dslr_res_mode",
        help="High-Speed Preview keeps sliders fast. Downloads will still compile at full resolution."
    )
    always_full_res = (resolution_mode == "Full Resolution (High Quality)")
    
    # Hash check configurations (Dual-Hash caching)
    # segment_hash: parameters affecting U²-Net/BiRefNet, Depth Anything V2, and Real-ESRGAN/GFPGAN enhancements
    segment_hash = (
        st.session_state.dslr_original_image,
        enhancement_mode,
        face_restoration,
        subject_mode,
        always_full_res
    )
    
    # blur_hash: parameters affecting spatially-varying DoF blending compositing
    blur_hash = (
        blur_mode,
        blur_preset,
        blur_strength,
        edge_feathering,
        subject_protection,
        background_smoothness
    )
    
    # ----------------- MAIN STUDIO INTERFACE -----------------
    if st.session_state.dslr_original_image is None:
        col_l, col_c, col_r = st.columns([1, 3, 1])
        with col_c:
            uploaded_img = render_upload_ui()
            if uploaded_img is not None:
                from PIL import ImageOps
                st.session_state.dslr_original_image = ImageOps.exif_transpose(uploaded_img)
                st.session_state.dslr_filename = "portrait"
                st.session_state.dslr_stage_cache = None
                st.session_state.dslr_preview_results = None
                st.session_state.dslr_full_res_results = None
                st.rerun()
                
            st.markdown("<p style='text-align: center; margin-top: 10px; color: #8A99AD;'>No files loaded? Test instantly with a pre-configured studio portrait:</p>", unsafe_allow_html=True)
            col_demo_left, col_demo_center, col_demo_right = st.columns([1, 1, 1])
            with col_demo_center:
                if st.button("Load Studio Demo", use_container_width=True):
                    load_demo_dslr_image()
                    st.rerun()
    else:
        col_title, col_reset = st.columns([5, 1])
        with col_reset:
            if st.button("Reset / Clear", key="dslr_clear", use_container_width=True):
                st.session_state.dslr_original_image = None
                st.session_state.dslr_filename = "portrait"
                st.session_state.dslr_stage_cache = None
                st.session_state.dslr_preview_results = None
                st.session_state.dslr_full_res_results = None
                st.rerun()
                
        # ----------------- DUAL-HASH STAGE 1: AI PHOTO ENHANCEMENT, SEGMENTATION, & DEPTH -----------------
        with st.spinner("Executing neural segmentation, face restoration, and depth estimation..."):
            
            if (
                st.session_state.dslr_stage_cache is None
                or st.session_state.dslr_stage_cache["hash"] != segment_hash
            ):
                max_dim = None if always_full_res else 800
                
                # 1. Run AI Photo Enhancement & Facial restoration (Real-ESRGAN + GFPGAN)
                enhanced_pil = PhotoEnhancer.process_enhancement(
                    pil_image=st.session_state.dslr_original_image,
                    mode=enhancement_mode,
                    face_restoration=face_restoration
                )
                
                # 2. Run high-fidelity subject segmentation (BiRefNet / U²-Net)
                seg_res = ImageProcessor.process_image(
                    pil_image=enhanced_pil,
                    rect=None,
                    margin_percentage=5.0,
                    iter_count=5,
                    closing_size=5,
                    keep_largest_only=True,
                    feather_radius=3,
                    matting_enabled=True,
                    shadow_enabled=False,
                    max_preview_dim=max_dim,
                    subject_mode=subject_mode
                )
                
                # Store cached segments
                st.session_state.dslr_stage_cache = {
                    "hash": segment_hash,
                    "enhanced": seg_res["original"], # Transposed & enhanced canvas
                    "mask": seg_res["mask"]           # Grayscale subject mask
                }
                
                # Invalidate subsequent blur preview composites
                st.session_state.dslr_preview_results = None
                st.session_state.dslr_full_res_results = None
                
            # Retrieve cached layers
            cached_enhanced = st.session_state.dslr_stage_cache["enhanced"]
            cached_mask = st.session_state.dslr_stage_cache["mask"]
            
        # ----------------- DUAL-HASH STAGE 2: SPATially-VARYING DEPTH BOKEH COMPOSITE -----------------
        with st.spinner("Synthesizing optical DSLR depth-of-field blur..."):
            
            if (
                st.session_state.dslr_preview_results is None
                or st.session_state.dslr_preview_results["hash"] != blur_hash
            ):
                # Execute Spatially Varying Depth Blur Engine
                blur_res = DepthBlurEngine.process_depth_blur(
                    pil_image=cached_enhanced,
                    mask_pil=cached_mask,
                    blur_mode=blur_mode,
                    blur_preset=blur_preset,
                    blur_strength=blur_strength,
                    edge_feathering=edge_feathering,
                    subject_protection=subject_protection,
                    background_smoothness=background_smoothness
                )
                
                st.session_state.dslr_preview_results = {
                    "hash": blur_hash,
                    "result": blur_res["result"],
                    "depth_map": blur_res["depth_map"]
                }
                
        preview_img = st.session_state.dslr_preview_results["result"]
        depth_img = st.session_state.dslr_preview_results["depth_map"]
        
        # ----------------- VISUAL STUDIO COMPARISONS -----------------
        st.markdown(
            """
            <div style="margin-top: 25px; margin-bottom: 15px;">
                <h3 style="color: #7000FF; font-weight: 600; letter-spacing: 0.5px; margin: 0 0 10px 0;">
                    🖼️ VISUAL COMPARISON
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Render a 3-column comparative view if depth map is previewed, otherwise 2 columns
        if show_depth_map:
            col_l, col_c, col_r = st.columns(3)
            with col_l:
                st.markdown('<p class="preview-label">ORIGINAL CANVASES</p>', unsafe_allow_html=True)
                st.image(st.session_state.dslr_original_image, use_container_width=True)
            with col_c:
                st.markdown('<p class="preview-label" style="color: #7000FF;">ESTIMATED RELATIVE DEPTH</p>', unsafe_allow_html=True)
                st.image(depth_img, use_container_width=True)
            with col_r:
                st.markdown('<p class="preview-label" style="color: #00F0FF;">DSLR PORTRAIT COMPOSITE</p>', unsafe_allow_html=True)
                st.image(preview_img, use_container_width=True)
        else:
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown('<p class="preview-label">ORIGINAL SOURCE IMAGE</p>', unsafe_allow_html=True)
                st.image(st.session_state.dslr_original_image, use_container_width=True)
            with col_right:
                st.markdown('<p class="preview-label" style="color: #7000FF;">PRO DSLR BOKEH COMPOSITE</p>', unsafe_allow_html=True)
                st.image(preview_img, use_container_width=True)
                
        # ----------------- LAZY HIGH-RESOLUTION COMPILER -----------------
        if not always_full_res:
            st.info("ℹ️ Export files are generated at full high-resolution. High-quality processing builds when clicking below.")
            
            # Setup full resolution hash keys
            full_seg_hash = (
                st.session_state.dslr_original_image,
                enhancement_mode,
                face_restoration,
                subject_mode,
                True # Full resolution
            )
            
            if (
                st.session_state.dslr_full_res_results is None
                or st.session_state.dslr_full_res_results["hash"] != (full_seg_hash, blur_hash)
            ):
                st.session_state.dslr_full_res_results = None # clean cache
                
                with st.spinner("Compiling full high-resolution DSLR portrait (Upscaling & Face Restoration)..."):
                    # 1. Full-Res Photo Enhancement
                    full_enhanced_pil = PhotoEnhancer.process_enhancement(
                        pil_image=st.session_state.dslr_original_image,
                        mode=enhancement_mode,
                        face_restoration=face_restoration
                    )
                    
                    # 2. Full-Res Subject Segmentation
                    full_seg_res = ImageProcessor.process_image(
                        pil_image=full_enhanced_pil,
                        rect=None,
                        margin_percentage=5.0,
                        iter_count=5,
                        closing_size=5,
                        keep_largest_only=True,
                        feather_radius=3,
                        matting_enabled=True,
                        shadow_enabled=False,
                        max_preview_dim=None, # Full size
                        subject_mode=subject_mode
                    )
                    
                    # 3. Full-Res Spatially Varying Depth Blur
                    full_blur_res = DepthBlurEngine.process_depth_blur(
                        pil_image=full_seg_res["original"],
                        mask_pil=full_seg_res["mask"],
                        blur_mode=blur_mode,
                        blur_preset=blur_preset,
                        blur_strength=blur_strength,
                        edge_feathering=edge_feathering,
                        subject_protection=subject_protection,
                        background_smoothness=background_smoothness
                    )
                    
                    st.session_state.dslr_full_res_results = {
                        "hash": (full_seg_hash, blur_hash),
                        "result": full_blur_res["result"]
                    }
                    
            dl_image = st.session_state.dslr_full_res_results["result"]
        else:
            dl_image = preview_img
            
        # ----------------- PRO PORTRAIT EXPORTS -----------------
        st.markdown(
            """
            <div style="margin-top: 30px; margin-bottom: 15px;">
                <h3 style="color: #00F0FF; font-weight: 600; letter-spacing: 0.5px; margin: 0 0 10px 0;">
                    💾 EXPORT PORTRAIT
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown('<div class="glass-card" style="padding: 20px;">', unsafe_allow_html=True)
        col_dl_jpg, col_dl_png = st.columns(2)
        
        filename_base = st.session_state.dslr_filename.rsplit('.', 1)[0]
        
        with col_dl_jpg:
            jpg_buf = io.BytesIO()
            dl_image.convert("RGB").save(jpg_buf, format="JPEG", quality=95)
            jpg_bytes = jpg_buf.getvalue()
            
            st.markdown(
                """
                <div style="text-align: center; margin-bottom: 10px;">
                    <span style="font-size: 1.8rem;">📸</span>
                    <p style="font-weight: 600; margin: 5px 0 0 0; color: #FFF;">High Quality JPG</p>
                    <p style="font-size: 0.75rem; color: #8A99AD; margin: 2px 0 10px 0;">Great for social sharing. Compression matches standard camera outputs.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.download_button(
                label="Download DSLR Portrait JPG",
                data=jpg_bytes,
                file_name=f"{filename_base}_dslr_portrait.jpg",
                mime="image/jpeg",
                key="dl_dslr_jpg",
                use_container_width=True
            )
            
        with col_dl_png:
            png_buf = io.BytesIO()
            dl_image.save(png_buf, format="PNG")
            png_bytes = png_buf.getvalue()
            
            st.markdown(
                """
                <div style="text-align: center; margin-bottom: 10px;">
                    <span style="font-size: 1.8rem;">✨</span>
                    <p style="font-weight: 600; margin: 5px 0 0 0; color: #FFF;">Lossless PNG</p>
                    <p style="font-size: 0.75rem; color: #8A99AD; margin: 2px 0 10px 0;">Lossless quality containing perfect sub-pixel detail and transparency.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.download_button(
                label="Download DSLR Portrait PNG",
                data=png_bytes,
                file_name=f"{filename_base}_dslr_portrait.png",
                mime="image/png",
                key="dl_dslr_png",
                use_container_width=True
            )
            
        st.markdown('</div>', unsafe_allow_html=True)
