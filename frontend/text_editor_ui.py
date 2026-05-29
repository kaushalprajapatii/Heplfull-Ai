import streamlit as st
from PIL import Image, ImageDraw
import os
import io

from backend.text_editor.pdf_processor import get_pdf_page_count, pdf_page_to_pil, compile_images_to_pdf
from backend.text_editor.ocr_engine import EASYOCR_AVAILABLE
from backend.text_editor.orchestrator import TextEditorOrchestrator
from backend.text_editor.font_detector import load_matching_font

def load_demo_invoice():
    """Create a beautiful synthetic demo invoice image to test instantly."""
    w, h = 700, 900
    img = Image.new("RGB", (w, h), "#FAFAFA")
    draw = ImageDraw.Draw(img)
    
    draw.rectangle([20, 20, w-20, h-20], fill="#FFFFFF", outline="#E0E0E0", width=2)
    
    try:
        from PIL import ImageFont
        font_large = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 36)
        font_med = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 16)
        font_bold = ImageFont.truetype("C:\\Windows\\Fonts\\arialbd.ttf", 16)
    except Exception:
        font_large = ImageFont.load_default()
        font_med = ImageFont.load_default()
        font_bold = ImageFont.load_default()
        
    draw.text((50, 60), "INVOICE", fill="#1A1B35", font=font_large)
    draw.text((50, 140), "Invoice Number: INV-2026-9042", fill="#555555", font=font_med)
    draw.text((50, 170), "Date: May 29, 2026", fill="#555555", font=font_med)
    draw.text((50, 200), "Billed To: Antigravity User", fill="#555555", font=font_med)
    
    draw.line([50, 240, w-50, 240], fill="#E0E0E0", width=1)
    
    draw.text((50, 260), "Description", fill="#1A1B35", font=font_bold)
    draw.text((450, 260), "Quantity", fill="#1A1B35", font=font_bold)
    draw.text((550, 260), "Amount", fill="#1A1B35", font=font_bold)
    
    draw.line([50, 290, w-50, 290], fill="#1A1B35", width=2)
    
    draw.text((50, 310), "AI Background Remover Software Licence", fill="#555555", font=font_med)
    draw.text((450, 310), "1", fill="#555555", font=font_med)
    draw.text((550, 310), "$149.00", fill="#555555", font=font_med)
    
    draw.text((50, 350), "Advanced Guided Filter Matting Module", fill="#555555", font=font_med)
    draw.text((450, 350), "2", fill="#555555", font=font_med)
    draw.text((550, 350), "$199.00", fill="#555555", font=font_med)
    
    draw.line([50, 420, w-50, 420], fill="#E0E0E0", width=1)
    
    draw.text((400, 450), "Subtotal:", fill="#555555", font=font_med)
    draw.text((550, 450), "$348.00", fill="#555555", font=font_med)
    
    draw.text((400, 480), "Tax (10%):", fill="#555555", font=font_med)
    draw.text((550, 480), "$34.80", fill="#555555", font=font_med)
    
    draw.text((400, 520), "Total Due:", fill="#1A1B35", font=font_bold)
    draw.text((550, 520), "$382.80", fill="#1A1B35", font=font_bold)
    
    draw.rectangle([50, 780, w-50, 840], fill="#F0F4F8", outline="#D0DCE5")
    draw.text((70, 795), "Thank you for your business! For queries contact support@antigravity.ai", fill="#4A5D6E", font=font_med)
    
    st.session_state.te_original_file = img
    st.session_state.te_file_type = "image"
    st.session_state.te_pdf_filename = "demo_invoice.png"
    st.session_state.te_original_image = img
    st.session_state.te_pdf_pages = [img]
    st.session_state.te_edited_image = None
    st.session_state.te_replacements = []
    st.session_state.te_ocr_results = None

def draw_ocr_overlays_with_badges(pil_img: Image.Image, ocr_results: list) -> Image.Image:
    """Overlays high-quality translucent bounding boxes and numbered circular badges over text areas."""
    overlay_img = pil_img.copy()
    draw = ImageDraw.Draw(overlay_img, "RGBA")
    
    try:
        from PIL import ImageFont
        font_badge = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 13)
    except Exception:
        font_badge = ImageFont.load_default()
        
    for idx, item in enumerate(ocr_results):
        x, y, w, h = item["bbox"]
        # Draw translucent cyan bounding box
        draw.rectangle([x, y, x + w, y + h], fill=(0, 240, 255, 15), outline=(0, 240, 255, 160), width=1)
        
        # Draw small solid pink badge at top-left corner of bounding box
        bx, by = x - 5, y - 5
        br = 9  # radius
        draw.ellipse([bx - br, by - br, bx + br, by + br], fill=(255, 0, 127, 240))
        
        # Center number in circle
        num_str = str(idx + 1)
        draw.text((bx - 4, by - 6), num_str, fill=(255, 255, 255, 255), font=font_badge)
        
    return overlay_img

def generate_font_preview_pil(text: str, font_name: str) -> Image.Image:
    """Generates a small visual PIL image showing the replacement text rendered in the Google Font."""
    w, h = 280, 50
    canvas = Image.new("RGB", (w, h), "#F8F9FA")
    draw = ImageDraw.Draw(canvas)
    
    try:
        font, font_size = load_matching_font(bbox_height=26, font_family=font_name, size_multiplier=1.0, text=text)
        draw.text((10, 12), text, fill=(26, 27, 53), font=font)
    except Exception:
        draw.text((10, 15), text, fill=(26, 27, 53))
        
    return canvas

def render_text_editor_ui():
    """Renders the modular AI In-Image & PDF Document Text Editor workspace with interactive numbered badges."""
    st.markdown(
        """
        <div style="margin-top: 10px; margin-bottom: 20px;">
            <h2 style="color: #FF007F; font-weight: 700; letter-spacing: -0.5px; margin: 0;">📝 AI Document & PDF Text Editor</h2>
            <p style="color: #8A99AD; font-size: 0.95rem; margin: 2px 0 0 0;">Correct text directly using visual numbered badges. Search and preview the most matching Google Fonts.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # ----------------- SESSION STATE INITIALIZATION -----------------
    if "te_original_file" not in st.session_state:
        st.session_state.te_original_file = None
    if "te_file_type" not in st.session_state:
        st.session_state.te_file_type = None
    if "te_pdf_filename" not in st.session_state:
        st.session_state.te_pdf_filename = "document"
    if "te_original_image" not in st.session_state:
        st.session_state.te_original_image = None
    if "te_pdf_pages" not in st.session_state:
        st.session_state.te_pdf_pages = []
    if "te_active_page" not in st.session_state:
        st.session_state.te_active_page = 0
    if "te_ocr_results" not in st.session_state:
        st.session_state.te_ocr_results = None
    if "te_edited_image" not in st.session_state:
        st.session_state.te_edited_image = None
    if "te_replacements" not in st.session_state:
        st.session_state.te_replacements = []
        
    # ----------------- SIDEBAR CONTROLS -----------------
    st.sidebar.markdown("### ⚙️ TEXT EDITOR CONTROLS")
    
    font_family = st.sidebar.selectbox(
        "Default Font Family",
        ["Sans-Serif", "Serif", "Monospace"],
        index=0,
        help="Selects the nearest matching system TTF font style for text rendering."
    )
    
    custom_font = st.sidebar.text_input(
        "⚡ Or Search Google Font",
        value="",
        placeholder="e.g. Rozha One, Kalam, Hind, Poppins",
        help="Type any Google Font family name. The engine will dynamically download, extract, and cache the full font set!"
    )
    
    if custom_font.strip():
        font_family = custom_font.strip()
        
    size_multiplier = st.sidebar.slider(
        "Font Size Scale Adjust",
        min_value=0.5,
        max_value=1.5,
        value=0.90,
        step=0.05,
        help="Scale multiplier to perfectly size the drawn characters to the surrounding bounding box."
    )
    
    overlay_ocr = st.sidebar.checkbox(
        "Overlay Numbered Badges",
        value=True,
        help="Draws glowing pink circular badges and cyan boxes over text blocks to make editing easy."
    )
    
    # ----------------- MAIN WORKFLOW -----------------
    
    # Case 1: No file loaded yet
    if st.session_state.te_original_file is None:
        col_l, col_c, col_r = st.columns([1, 3, 1])
        with col_c:
            st.markdown(
                """
                <div class="glass-card upload-container">
                    <h3 style="text-align: center; color: #FF007F; margin-top: 0; font-weight: 600; letter-spacing: 0.5px;">
                        ⚡ UPLOAD SOURCE DOCUMENT
                    </h3>
                    <p style="text-align: center; color: #8A99AD; font-size: 0.9rem; margin-bottom: 20px;">
                        Upload an Image or multi-page PDF Document. Coordinates and fonts will be parsed instantly.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            uploaded_file = st.file_uploader(
                "Choose document or image...",
                type=["pdf", "jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
                key="doc_uploader"
            )
            
            if uploaded_file is not None:
                try:
                    file_bytes = uploaded_file.read()
                    filename = uploaded_file.name
                    st.session_state.te_pdf_filename = filename
                    
                    if filename.lower().endswith(".pdf"):
                        st.session_state.te_original_file = file_bytes
                        st.session_state.te_file_type = "pdf"
                        page_count = get_pdf_page_count(file_bytes)
                        
                        with st.spinner(f"Extracting {page_count} PDF pages at high quality (150 DPI)..."):
                            pdf_pages = []
                            for idx in range(page_count):
                                pdf_pages.append(pdf_page_to_pil(file_bytes, idx, dpi=150))
                            st.session_state.te_pdf_pages = pdf_pages
                            st.session_state.te_active_page = 0
                            st.session_state.te_original_image = pdf_pages[0]
                    else:
                        from PIL import ImageOps
                        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                        pil_img = ImageOps.exif_transpose(pil_img)
                        st.session_state.te_original_file = file_bytes
                        st.session_state.te_file_type = "image"
                        st.session_state.te_pdf_pages = [pil_img]
                        st.session_state.te_active_page = 0
                        st.session_state.te_original_image = pil_img
                        
                    st.session_state.te_edited_image = None
                    st.session_state.te_replacements = []
                    st.session_state.te_ocr_results = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading document: {str(e)}")
                    
            st.markdown("<p style='text-align: center; margin-top: 10px; color: #8A99AD;'>No files ready? Test instantly with a pre-configured demo invoice:</p>", unsafe_allow_html=True)
            col_demo_left, col_demo_center, col_demo_right = st.columns([1, 1, 1])
            with col_demo_center:
                if st.button("Load Demo Invoice", use_container_width=True):
                    load_demo_invoice()
                    st.rerun()
                    
    # Case 2: Document loaded
    else:
        col_nav, col_reset = st.columns([5, 1])
        
        with col_reset:
            if st.button("Reset / Clear", key="te_reset", use_container_width=True):
                st.session_state.te_original_file = None
                st.session_state.te_file_type = None
                st.session_state.te_pdf_filename = "document"
                st.session_state.te_original_image = None
                st.session_state.te_pdf_pages = []
                st.session_state.te_ocr_results = None
                st.session_state.te_edited_image = None
                st.session_state.te_replacements = []
                st.rerun()
                
        with col_nav:
            if st.session_state.te_file_type == "pdf":
                pages_count = len(st.session_state.te_pdf_pages)
                st.markdown(
                    f"""
                    <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px 15px; margin-bottom: 15px;">
                        <span>📄 File: <b>{st.session_state.te_pdf_filename}</b> | Total Pages: <b>{pages_count}</b></span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if pages_count > 1:
                    active_pg = st.slider("Navigate Pages", 1, pages_count, st.session_state.te_active_page + 1) - 1
                    if active_pg != st.session_state.te_active_page:
                        st.session_state.te_active_page = active_pg
                        st.session_state.te_original_image = st.session_state.te_pdf_pages[active_pg]
                        st.session_state.te_ocr_results = None
                        st.session_state.te_edited_image = None
                        st.session_state.te_replacements = []
                        st.rerun()
            else:
                st.markdown(
                    f"""
                    <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px 15px; margin-bottom: 15px;">
                        <span>🖼️ File: <b>{st.session_state.te_pdf_filename}</b> ({st.session_state.te_original_image.width}x{st.session_state.te_original_image.height}px)</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        # ----------------- OCR BACKGROUND TASK -----------------
        if EASYOCR_AVAILABLE and st.session_state.te_ocr_results is None:
            with st.spinner("Analyzing image text layout using OCR engine..."):
                ocr_res = TextEditorOrchestrator.run_ocr(st.session_state.te_original_image)
                st.session_state.te_ocr_results = ocr_res
                
        # ----------------- DUAL COLUMN LAYOUT -----------------
        col_preview, col_editor_panel = st.columns([1, 1])
        
        # Left Column: Image Previews (Before/After)
        with col_preview:
            st.markdown('<p class="preview-label">1. ORIGINAL SOURCE DOCUMENT</p>', unsafe_allow_html=True)
            
            # Display image with circular number badges
            if overlay_ocr and st.session_state.te_ocr_results:
                original_badged = draw_ocr_overlays_with_badges(st.session_state.te_original_image, st.session_state.te_ocr_results)
                st.image(original_badged, use_container_width=True)
            else:
                st.image(st.session_state.te_original_image, use_container_width=True)
                
            st.markdown('<p class="preview-label" style="margin-top:20px;">2. EDITED DOCUMENT PREVIEW</p>', unsafe_allow_html=True)
            if st.session_state.te_edited_image is not None:
                st.image(st.session_state.te_edited_image, use_container_width=True)
            else:
                st.markdown(
                    """
                    <div style="display: flex; align-items: center; justify-content: center; height: 300px; background: rgba(255, 255, 255, 0.02); border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 8px;">
                        <p style="color: #8A99AD; font-size: 0.95rem; text-align: center;">
                            No edits made yet.<br>Modify a text block on the right panel to see updates.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        # Right Column: Visual In-Place Editor Panel
        with col_editor_panel:
            st.markdown('<p class="preview-label" style="color: #FF007F;">3. CLICK & EDIT DOCUMENT TEXT FIELDS</p>', unsafe_allow_html=True)
            
            if not st.session_state.te_ocr_results:
                st.warning("⚠️ No text detected in this page. Try uploading an image with higher contrast or clear characters.")
            else:
                # Present list of all detected blocks as visual numbered rows
                st.markdown('<div class="glass-card" style="padding: 15px; max-height: 520px; overflow-y: scroll; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 20px;">', unsafe_allow_html=True)
                
                # Dictionary to store user's changed inputs
                user_replacements = {}
                
                for idx, item in enumerate(st.session_state.te_ocr_results):
                    # Bounding Box description row
                    orig_text = item["text"]
                    bbox = item["bbox"]
                    
                    st.markdown(
                        f"""
                        <div style="display: flex; align-items: center; margin-top: 10px; margin-bottom: 3px;">
                            <span style="background: #FF007F; color: white; border-radius: 50%; width: 22px; height: 22px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: bold; margin-right: 8px;">
                                {idx+1}
                            </span>
                            <span style="font-size: 0.8rem; color: #8A99AD; text-transform: uppercase; font-weight: bold; letter-spacing: 0.5px;">Original:</span>
                            <span style="font-size: 0.9rem; color: #E2E8F0; font-weight: 500; margin-left: 5px;">"{orig_text}"</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Edit input box
                    rep_input = st.text_input(
                        "Replacement Value",
                        value=orig_text,
                        key=f"rep_val_{idx}",
                        label_visibility="collapsed"
                    )
                    
                    # Store if the user actually edited it
                    if rep_input != orig_text:
                        user_replacements[idx] = {
                            "bbox": bbox,
                            "original_text": orig_text,
                            "replacement_text": rep_input
                        }
                        
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Apply Corrections Button
                col_btn_l, col_btn_c = st.columns([3, 1])
                with col_btn_c:
                    apply_batch = st.button("Apply Corrections", use_container_width=True)
                    
                # ----------------- DYNAMIC GOOGLE FONT MATCHING GALLERY -----------------
                st.markdown('<p class="preview-label" style="color: #00F0FF; margin-top: 25px;">4. MATCHING GOOGLE FONT PREVIEW GALLERY</p>', unsafe_allow_html=True)
                st.info("💡 **Google Font Search**: Choose a text block on the left and see how it renders in these popular Google Fonts in real-time, then type the matching name into **Or Search Google Font** in the sidebar!")
                
                # Let user pick which replacement string to preview in the gallery
                words_to_preview = [item["text"] for item in st.session_state.te_ocr_results]
                selected_preview_word = st.selectbox("Select word to preview fonts:", words_to_preview)
                
                # Font Showcase grid
                st.markdown('<div class="glass-card" style="padding: 15px;">', unsafe_allow_html=True)
                
                hindi_showcase_fonts = {
                    "Rozha One": "Thick Display Serifs (Perfect for headers)",
                    "Noto Sans Devanagari": "Sleek Modern Sans-Serif",
                    "Yatra One": "Classic Indian Calligraphy",
                    "Kalam": "Handwritten/Artistic Calligraphy",
                    "Hind": "Clean corporate Sans-Serif",
                    "Martel": "Formal Document Serif",
                    "Teko": "Condensed Bold Display"
                }
                
                # Display 3 popular fonts previews side-by-side
                col_f1, col_f2 = st.columns(2)
                
                font_list = list(hindi_showcase_fonts.keys())
                for i, font_name in enumerate(font_list):
                    col_target = col_f1 if i % 2 == 0 else col_f2
                    with col_target:
                        st.markdown(f"**{font_name}** (*{hindi_showcase_fonts[font_name]}*)")
                        preview_pil = generate_font_preview_pil(selected_preview_word, font_name)
                        st.image(preview_pil, use_container_width=True)
                        st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)
                        
                st.markdown('</div>', unsafe_allow_html=True)
                
                # ----------------- RUN CORRECTIONS -----------------
                if apply_batch:
                    if not user_replacements:
                        st.warning("⚠️ No changes were made in any text field inputs. Modify a text box first!")
                    else:
                        with st.spinner("Applying all visual text corrections..."):
                            canvas = st.session_state.te_original_image.copy()
                            
                            # Apply each replacement in order
                            for idx in sorted(user_replacements.keys()):
                                item = user_replacements[idx]
                                bbox = item["bbox"]
                                new_text = item["replacement_text"]
                                
                                rep_instruction = {
                                    "bbox": bbox,
                                    "replacement_text": new_text
                                }
                                
                                canvas = TextEditorOrchestrator.apply_replacements(
                                    pil_image=canvas,
                                    replacements=[rep_instruction],
                                    font_family=font_family,
                                    size_multiplier=size_multiplier
                                )
                                
                                st.session_state.te_replacements.append({
                                    "original": item["original_text"],
                                    "replacement": new_text,
                                    "bbox": bbox
                                })
                                
                            st.session_state.te_edited_image = canvas
                            st.session_state.te_pdf_pages[st.session_state.te_active_page] = canvas
                            st.toast("Success! All selected text blocks updated.", icon="✨")
                            st.rerun()
                            
        # ----------------- DOWNLOAD COMPLETED DOCUMENT -----------------
        if st.session_state.te_edited_image is not None or len(st.session_state.te_replacements) > 0:
            st.markdown(
                """
                <div style="margin-top: 30px; margin-bottom: 15px;">
                    <h3 style="color: #00F0FF; font-weight: 600; letter-spacing: 0.5px; margin: 0 0 10px 0;">
                        💾 EXPORT DOCUMENT
                    </h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown('<div class="glass-card" style="padding: 20px;">', unsafe_allow_html=True)
            
            filename_base = st.session_state.te_pdf_filename.rsplit('.', 1)[0]
            
            if st.session_state.te_file_type == "pdf":
                col_dl_pdf, col_dl_cur = st.columns(2)
                
                with col_dl_pdf:
                    with st.spinner("Compiling all multi-page PDF pages..."):
                        compiled_pdf_bytes = compile_images_to_pdf(st.session_state.te_pdf_pages)
                        
                    st.markdown(
                        """
                        <div style="text-align: center; margin-bottom: 10px;">
                            <span style="font-size: 1.8rem;">📄</span>
                            <p style="font-weight: 600; margin: 5px 0 0 0; color: #FFF;">Complete Edited PDF</p>
                            <p style="font-size: 0.75rem; color: #8A99AD; margin: 2px 0 10px 0;">Re-compiles all pages into a single PDF document.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    st.download_button(
                        label="Download Full Document PDF",
                        data=compiled_pdf_bytes,
                        file_name=f"{filename_base}_edited.pdf",
                        mime="application/pdf",
                        key="dl_edited_pdf",
                        use_container_width=True
                    )
                    
                with col_dl_cur:
                    pg_buf = io.BytesIO()
                    st.session_state.te_edited_image.save(pg_buf, format="PNG")
                    pg_bytes = pg_buf.getvalue()
                    
                    st.markdown(
                        """
                        <div style="text-align: center; margin-bottom: 10px;">
                            <span style="font-size: 1.8rem;">🖼️</span>
                            <p style="font-weight: 600; margin: 5px 0 0 0; color: #FFF;">Active Page Image (PNG)</p>
                            <p style="font-size: 0.75rem; color: #8A99AD; margin: 2px 0 10px 0;">Downloads the current page as a single high-DPI image.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    st.download_button(
                        label=f"Download Page {st.session_state.te_active_page+1} PNG",
                        data=pg_bytes,
                        file_name=f"{filename_base}_page_{st.session_state.te_active_page+1}_edited.png",
                        mime="image/png",
                        key="dl_edited_page_png",
                        use_container_width=True
                    )
            else:
                img_buf = io.BytesIO()
                st.session_state.te_edited_image.save(img_buf, format="PNG")
                img_bytes = img_buf.getvalue()
                
                col_dl_img_l, col_dl_img_c, col_dl_img_r = st.columns([1, 2, 1])
                with col_dl_img_c:
                    st.markdown(
                        """
                        <div style="text-align: center; margin-bottom: 10px;">
                            <span style="font-size: 1.8rem;">✨</span>
                            <p style="font-weight: 600; margin: 5px 0 0 0; color: #FFF;">Edited Image Output</p>
                            <p style="font-size: 0.75rem; color: #8A99AD; margin: 2px 0 10px 0;">Downloads the text-corrected image as high-fidelity PNG.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    st.download_button(
                        label="Download Edited Image PNG",
                        data=img_bytes,
                        file_name=f"{filename_base}_edited.png",
                        mime="image/png",
                        key="dl_edited_img_png",
                        use_container_width=True
                    )
                    
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("#### 📜 REPLACEMENT LOGS")
            log_content = ""
            for i, rep in enumerate(st.session_state.te_replacements):
                log_content += f"{i+1}. Erased **'{rep['original']}'** and wrote **'{rep['replacement']}'** at bounding box `{rep['bbox']}`.\n"
            st.markdown(log_content)
