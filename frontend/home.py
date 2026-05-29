import streamlit as st

def render_sidebar_brand():
    """Renders a beautiful brand header in the sidebar."""
    st.sidebar.markdown(
        """
        <div style="text-align: center; margin-bottom: 25px; padding-bottom: 15px; border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
            <h2 style="margin: 0; background: linear-gradient(135deg, #00F0FF 0%, #FF007F 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 1.6rem; letter-spacing: -0.5px;">
                ⚡ ANTIGRAVITY STUDIO
            </h2>
            <p style="margin: 3px 0 0 0; color: #8A99AD; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px;">
                AI Image & Doc Workspace
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_home_dashboard():
    """Renders a stunning Glassmorphic landing dashboard listing both AI workspaces."""
    st.markdown(
        """
        <div style="text-align: center; margin-top: 30px; margin-bottom: 40px;">
            <h1 class="gradient-title" style="font-size: 3.5rem;">CREATIVE AI STUDIO</h1>
            <p class="subtitle" style="font-size: 1.25rem;">
                Powerhouse tools for automated high-fidelity background removal and smart document text replacement.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div class="glass-card" style="height: 410px; display: flex; flex-direction: column; justify-content: space-between; border-top: 3px solid #00F0FF;">
                <div>
                    <span style="font-size: 3rem;">🖼️</span>
                    <h3 style="color: #00F0FF; margin-top: 15px; margin-bottom: 10px; font-weight: 700;">AI Background Remover</h3>
                    <p style="color: #8A99AD; font-size: 0.95rem; line-height: 1.6;">
                        Isolate subjects with pinpoint precision using U²-Net deep learning or general GrabCut extraction. 
                        Refine edge feathering, capture fine hair fibers with Guided Filter matting, and compose realistic 
                        drop shadows.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Launch Background Remover", key="launch_bg", use_container_width=True):
            st.session_state.active_workspace = "AI Background Remover"
            st.rerun()
            
    with col2:
        st.markdown(
            """
            <div class="glass-card" style="height: 410px; display: flex; flex-direction: column; justify-content: space-between; border-top: 3px solid #7000FF;">
                <div>
                    <span style="font-size: 3rem;">📸</span>
                    <h3 style="color: #7000FF; margin-top: 15px; margin-bottom: 10px; font-weight: 700;">AI DSLR Portrait Blur</h3>
                    <p style="color: #8A99AD; font-size: 0.95rem; line-height: 1.6;">
                        Create beautiful portrait photographs with optical shallow depth-of-field. 
                        Features standard Gaussian blurs or realistic circular lens bokeh apertures with edge inpainting to 
                        completely eliminate colored halo bleeding.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Launch DSLR Portrait Blur", key="launch_blur", use_container_width=True):
            st.session_state.active_workspace = "AI DSLR Background Blur"
            st.rerun()
            
    with col3:
        st.markdown(
            """
            <div class="glass-card" style="height: 410px; display: flex; flex-direction: column; justify-content: space-between; border-top: 3px solid #FF007F;">
                <div>
                    <span style="font-size: 3rem;">📝</span>
                    <h3 style="color: #FF007F; margin-top: 15px; margin-bottom: 10px; font-weight: 700;">AI In-Image Text Editor</h3>
                    <p style="color: #8A99AD; font-size: 0.95rem; line-height: 1.6;">
                        Erase and replace text in images or multi-page PDFs seamlessly. 
                        Automatically scans fonts and text placement via EasyOCR, erases targeted ink using 
                        Telea texture-preserving inpainting, and draws new text in matching fonts, sizes, and colors.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Launch Document Text Editor", key="launch_te", use_container_width=True):
            st.session_state.active_workspace = "AI In-Image Text Editor"
            st.rerun()
            
    # Premium bottom status panel
    st.markdown(
        """
        <div style="margin-top: 50px; text-align: center; border-top: 1px solid rgba(255, 255, 255, 0.05); padding-top: 20px;">
            <p style="color: #8A99AD; font-size: 0.8rem;">
                ⚡ Powered by <b>U²-Net</b> Segmenter, <b>EasyOCR</b> Reader, <b>Guided Filter</b> Matting, <b>OpenCV</b> DSLR Blur & Inpainting.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
