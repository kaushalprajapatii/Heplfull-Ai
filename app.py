import streamlit as st
import os
import sys

# Add current workspace directory to sys.path to allow absolute imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from frontend.home import render_sidebar_brand, render_home_dashboard
from frontend.bg_remover_ui import render_bg_remover_ui
from frontend.text_editor_ui import render_text_editor_ui
from frontend.dslr_blur_ui import render_dslr_blur_ui

# Set Streamlit page config
st.set_page_config(
    page_title="Antigravity Studio - AI Image & Document Workspace",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Glassmorphic Dark Theme with glowing neon accents
st.markdown(
    """
    <style>
    /* Import modern Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Apply fonts and background styles */
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
        color: #E2E8F0;
    }
    
    /* Header Custom Styles */
    .gradient-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(135deg, #00F0FF 0%, #FF007F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }
    
    .subtitle {
        text-align: center;
        color: #8A99AD;
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 30px;
    }
    
    /* Glassmorphism Containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }
    
    /* Upload UI wrapper */
    .upload-container {
        padding: 40px;
        background: rgba(0, 240, 255, 0.01);
        border: 1px dashed rgba(0, 240, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        border-color: #FF007F;
        background: rgba(255, 0, 127, 0.01);
    }
    
    /* Custom Success Badge */
    .success-badge {
        background: rgba(0, 240, 255, 0.1);
        color: #00F0FF;
        border: 1px solid rgba(0, 240, 255, 0.2);
        border-radius: 8px;
        padding: 10px 15px;
        font-size: 0.9rem;
        margin: 15px 0;
        text-align: center;
    }
    
    /* Labels and Headers */
    .preview-label {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #FF007F;
        margin-bottom: 10px;
        text-align: center;
        text-transform: uppercase;
    }
    
    /* Custom Styled Sliders & Widgets in Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0A0D14;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    [data-testid="stSidebar"] .stMarkdown h2, [data-testid="stSidebar"] .stMarkdown h3 {
        color: #00F0FF;
        font-weight: 600;
    }
    
    /* Streamlit buttons customized */
    div.stButton > button {
        background: linear-gradient(135deg, #00F0FF 0%, #7000FF 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 15px rgba(0, 240, 255, 0.2);
    }
    
    div.stButton > button:hover {
        background: linear-gradient(135deg, #FF007F 0%, #7000FF 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 0, 127, 0.35);
        color: white;
    }
    
    /* Glowing accents */
    .glow-accent {
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.5);
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- SESSION STATE INITIALIZATION -----------------
if "active_workspace" not in st.session_state:
    st.session_state.active_workspace = "Home"

# ----------------- SIDEBAR BRANDING & ROUTING -----------------
render_sidebar_brand()

st.sidebar.markdown("### 🧭 NAVIGATION")
nav_choice = st.sidebar.selectbox(
    "Active Workspace",
    ["Home Dashboard", "AI Background Remover", "AI DSLR Background Blur", "AI In-Image Text Editor"],
    index=["Home Dashboard", "AI Background Remover", "AI DSLR Background Blur", "AI In-Image Text Editor"].index(
        "Home Dashboard" if st.session_state.active_workspace == "Home" else st.session_state.active_workspace
    ),
    key="nav_choice_select"
)

# Sync sidebar navigation selection with active session state
choice_mapped = "Home" if nav_choice == "Home Dashboard" else nav_choice
if choice_mapped != st.session_state.active_workspace:
    st.session_state.active_workspace = choice_mapped
    st.rerun()

# Quick nav back to Home from sidebar footer
st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
if st.session_state.active_workspace != "Home":
    if st.sidebar.button("← Back to Dashboard", use_container_width=True):
        st.session_state.active_workspace = "Home"
        st.rerun()

# ----------------- WORKSPACE RENDERING -----------------
if st.session_state.active_workspace == "Home":
    render_home_dashboard()
elif st.session_state.active_workspace == "AI Background Remover":
    render_bg_remover_ui()
elif st.session_state.active_workspace == "AI DSLR Background Blur":
    render_dslr_blur_ui()
elif st.session_state.active_workspace == "AI In-Image Text Editor":
    render_text_editor_ui()
