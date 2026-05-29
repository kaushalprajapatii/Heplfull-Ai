import streamlit as st
from PIL import Image
from backend.bg_remover.image_processor import ImageProcessor
from frontend.upload_ui import render_upload_ui
from frontend.preview_ui import render_preview_ui
from frontend.download_ui import render_download_ui

def load_demo_image():
    """Create a high-quality synthetic demo image (a solid red sphere with a smooth drop shadow on white background)."""
    w, h = 600, 600
    img = Image.new("RGB", (w, h), "#FFFFFF")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    cx, cy, r = 300, 300, 160
    for i in range(r, 0, -1):
        f = i / r
        cr = int(255 * (1.0 - f * 0.3))
        cg = int(0 * f)
        cb = int(128 + f * 127)
        draw.ellipse([cx - i, cy - i, cx + i, cy + i], fill=(cr, cg, cb))
        
    st.session_state.original_image = img
    st.session_state.filename = "demo_gradient_sphere.png"
    st.session_state.processed_results = None  # clear previous cache

def render_bg_remover_ui():
    """Renders the complete AI Background Remover workspace, including sidebar controls and visual output areas."""
    st.markdown(
        """
        <div style="margin-top: 10px; margin-bottom: 20px;">
            <h2 style="color: #00F0FF; font-weight: 700; letter-spacing: -0.5px; margin: 0;">🖼️ AI Background Remover</h2>
            <p style="color: #8A99AD; font-size: 0.95rem; margin: 2px 0 0 0;">Extract foreground subjects, refine hair details, and generate dropshadows.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # ----------------- SESSION STATE INITIALIZATION -----------------
    if "original_image" not in st.session_state:
        st.session_state.original_image = None
    if "filename" not in st.session_state:
        st.session_state.filename = "image"
    if "processed_results" not in st.session_state:
        st.session_state.processed_results = None
        
    # ----------------- SIDEBAR CONTROLS -----------------
    st.sidebar.markdown("### ⚙️ BG REMOVER CONTROLS")
    
    subject_mode = st.sidebar.selectbox(
        "Subject Extraction Mode",
        ["AI BiRefNet (SOTA General)", "AI U²-Net (Legacy Neural)", "General Subject (GrabCut)", "Signature & Text (Ink)"],
        index=0,
        help="AI BiRefNet uses state-of-the-art bilateral reference models for ultra-sharp edge matting. U²-Net is legacy neural. General uses GrabCut margins. Signature extracts ink."
    )
    
    if subject_mode == "General Subject (GrabCut)":
        st.sidebar.markdown("#### 🟥 Bounding Box Configuration")
        bb_mode = st.sidebar.radio(
            "Initialization Mode",
            ["Automated Bounding Box (Recommended)", "Manual Coordinate Sliders"],
            help="Automated finds the subject using edge density. Manual lets you control the crop boundary."
        )
        
        manual_bbox = None
        if bb_mode == "Manual Coordinate Sliders" and st.session_state.original_image is not None:
            img_w, img_h = st.session_state.original_image.size
            st.sidebar.info("Adjust margins in pixels to frame the subject:")
            bb_x = st.sidebar.slider("Left Coordinate (X)", 0, img_w - 20, int(img_w * 0.05))
            bb_y = st.sidebar.slider("Top Coordinate (Y)", 0, img_h - 20, int(img_h * 0.05))
            bb_w = st.sidebar.slider("Width (W)", 20, img_w - bb_x, int(img_w * 0.9))
            bb_h = st.sidebar.slider("Height (H)", 20, img_h - bb_y, int(img_h * 0.9))
            manual_bbox = (bb_x, bb_y, bb_w, bb_h)
            
        iter_count = st.sidebar.slider(
            "GrabCut Iterations", 
            min_value=1, 
            max_value=10, 
            value=5, 
            help="Higher values refine the classification but process slower."
        )
        
        bg_seed_sensitivity = st.sidebar.slider(
            "Background Color Seeding",
            min_value=0.0,
            max_value=100.0,
            value=35.0,
            step=1.0,
            help="Assists GrabCut by matching background colors from image corners. Set to 0.0 to disable seeding."
        )
        show_bb = (bb_mode == "Manual Coordinate Sliders")
    else:
        manual_bbox = None
        iter_count = 5
        bg_seed_sensitivity = 35.0
        show_bb = False
        bb_mode = "Automated Bounding Box (Recommended)"
        
    st.sidebar.markdown("#### ⚡ Edge Refinement & Softening")
    closing_size = st.sidebar.slider(
        "Morphological Closing Size", 
        min_value=1, 
        max_value=21, 
        value=5, 
        step=2, 
        help="Fills holes and smooths the interior contours of the cutout."
    )
    feather_radius = st.sidebar.slider(
        "Edge Feathering Radius", 
        min_value=0, 
        max_value=15, 
        value=3, 
        help="Creates smooth anti-aliased edge blends. Prevents pixelation."
    )
    keep_largest = st.sidebar.checkbox(
        "Remove Stray Speckles", 
        value=True, 
        help="Keeps only the largest foreground object, filtering out background noise."
    )
    
    matting_enabled = st.sidebar.checkbox(
        "Enable Hair & Detail Matting",
        value=True,
        help="Uses Guided Filter to align cutout edges perfectly to fine hair strands."
    )
    
    if matting_enabled:
        matting_radius = st.sidebar.slider(
            "Matting Search Radius",
            min_value=1,
            max_value=30,
            value=10,
            help="Search window radius for hair and fine details."
        )
        matting_eps = st.sidebar.slider(
            "Matting Smoothness (Eps)",
            min_value=0.0001,
            max_value=0.01,
            value=0.001,
            step=0.0005,
            format="%.4f",
            help="Controls details alignment. Lower values capture sharper hair details."
        )
    else:
        matting_radius = 10
        matting_eps = 0.001
        
    st.sidebar.markdown("#### 🌗 Realistic Shadow Engine")
    shadow_enabled = st.sidebar.checkbox("Enable Drop Shadow", value=False)
    
    if shadow_enabled:
        shadow_opacity = st.sidebar.slider("Shadow Opacity", 0.0, 1.0, 0.4, 0.05)
        shadow_blur = st.sidebar.slider("Shadow Softness (Blur)", 0, 50, 15, 1)
        shadow_distance = st.sidebar.slider("Shadow Distance Offset", 0, 100, 20, 1)
        shadow_angle = st.sidebar.slider("Shadow Direction (Angle)", 0, 360, 45, 5, format="%d°")
    else:
        shadow_opacity, shadow_blur, shadow_distance, shadow_angle = 0.4, 15, 20, 45
        
    st.sidebar.markdown("#### 🚀 Performance Options")
    resolution_mode = st.sidebar.radio(
        "Processing Resolution",
        ["High-Speed Preview (Max 800px)", "Full Resolution (High Quality)"],
        help="High-Speed Preview keeps UI sliders highly responsive. When downloaded, images can run at full size."
    )
    always_full_res = (resolution_mode == "Full Resolution (High Quality)")
    
    # Hash check
    param_hash = (
        manual_bbox,
        iter_count,
        bg_seed_sensitivity,
        closing_size,
        feather_radius,
        keep_largest,
        matting_enabled,
        matting_radius,
        matting_eps,
        shadow_enabled,
        shadow_opacity,
        shadow_blur,
        shadow_distance,
        shadow_angle,
        always_full_res,
        subject_mode
    )
    
    # ----------------- MAIN CONTENT AREA -----------------
    if st.session_state.original_image is None:
        col_l, col_c, col_r = st.columns([1, 3, 1])
        with col_c:
            uploaded_img = render_upload_ui()
            if uploaded_img is not None:
                st.session_state.original_image = uploaded_img
                st.session_state.processed_results = None
                st.rerun()
                
            st.markdown("<p style='text-align: center; margin-top: 10px; color: #8A99AD;'>No image ready? Test instantly with a pre-configured demo image:</p>", unsafe_allow_html=True)
            col_demo_left, col_demo_center, col_demo_right = st.columns([1, 1, 1])
            with col_demo_center:
                if st.button("Load Demo Sphere", use_container_width=True):
                    load_demo_image()
                    st.rerun()
    else:
        col_title, col_reset = st.columns([5, 1])
        with col_reset:
            if st.button("Reset / Clear", use_container_width=True):
                st.session_state.original_image = None
                st.session_state.filename = "image"
                st.session_state.processed_results = None
                st.rerun()
                
        with st.spinner("Processing foreground extraction pipeline..."):
            if (
                st.session_state.processed_results is None 
                or st.session_state.processed_results.get("param_hash") != param_hash
            ):
                max_dim = None if always_full_res else 800
                
                results = ImageProcessor.process_image(
                    pil_image=st.session_state.original_image,
                    rect=manual_bbox,
                    margin_percentage=5.0,
                    iter_count=iter_count,
                    bg_seed_sensitivity=bg_seed_sensitivity,
                    closing_size=closing_size,
                    keep_largest_only=keep_largest,
                    feather_radius=feather_radius,
                    matting_enabled=matting_enabled,
                    matting_radius=matting_radius,
                    matting_eps=matting_eps,
                    shadow_enabled=shadow_enabled,
                    shadow_opacity=shadow_opacity,
                    shadow_blur=shadow_blur,
                    shadow_distance=shadow_distance,
                    shadow_angle=shadow_angle,
                    max_preview_dim=max_dim,
                    subject_mode=subject_mode
                )
                results["param_hash"] = param_hash
                st.session_state.processed_results = results
                
        results = st.session_state.processed_results
        
        if keep_largest:
            st.info("💡 **Working with Signatures or Text?** If parts of your signature (like separate letters or strokes) are missing, **uncheck 'Remove Stray Speckles'** in the sidebar to prevent them from being filtered out.")
            
        render_preview_ui(
            original_pil=results["original"],
            processed_pil=results["shadow"] if shadow_enabled else results["transparent"],
            bounding_box=results["rect"],
            show_bounding_box=show_bb
        )
        
        # Compile full resolution downloads lazy
        if not always_full_res:
            st.info("ℹ️ Download files are generated at full high-resolution. Processing begins when clicking below.")
            
            if "full_res_results" not in st.session_state or st.session_state.get("full_res_hash") != param_hash:
                with st.spinner("Compiling full high-resolution outputs..."):
                    full_res = ImageProcessor.process_image(
                        pil_image=st.session_state.original_image,
                        rect=manual_bbox,
                        margin_percentage=5.0,
                        iter_count=iter_count,
                        bg_seed_sensitivity=bg_seed_sensitivity,
                        closing_size=closing_size,
                        keep_largest_only=keep_largest,
                        feather_radius=feather_radius,
                        matting_enabled=matting_enabled,
                        matting_radius=matting_radius,
                        matting_eps=matting_eps,
                        shadow_enabled=shadow_enabled,
                        shadow_opacity=shadow_opacity,
                        shadow_blur=shadow_blur,
                        shadow_distance=shadow_distance,
                        shadow_angle=shadow_angle,
                        max_preview_dim=None,  # Full resolution
                        subject_mode=subject_mode
                    )
                    st.session_state.full_res_results = full_res
                    st.session_state.full_res_hash = param_hash
                    
            dl_transparent = st.session_state.full_res_results["transparent"]
            dl_shadow = st.session_state.full_res_results["shadow"]
        else:
            dl_transparent = results["transparent"]
            dl_shadow = results["shadow"]
            
        render_download_ui(
            transparent_pil=dl_transparent,
            shadow_pil=dl_shadow,
            filename=st.session_state.filename
        )
