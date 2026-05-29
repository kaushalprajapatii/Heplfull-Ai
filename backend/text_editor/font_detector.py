import os
from PIL import ImageFont
from backend.text_editor.font_downloader import download_google_font

def get_system_font_path(font_family: str = "Sans-Serif") -> str:
    """
    Get the path to a standard system font on Windows.
    Defaults to Arial (Sans-Serif), Times New Roman (Serif), or Courier New (Monospace).
    """
    windows_font_dir = "C:\\Windows\\Fonts"
    
    font_mapping = {
        "Sans-Serif": ["arial.ttf", "calibri.ttf", "segoeui.ttf"],
        "Serif": ["times.ttf", "georgia.ttf", "cambria.ttf"],
        "Monospace": ["cour.ttf", "consola.ttf", "lucon.ttf"]
    }
    
    candidates = font_mapping.get(font_family, font_mapping["Sans-Serif"])
    
    # Check if we are on Windows and directory exists
    if os.path.exists(windows_font_dir):
        for candidate in candidates:
            path = os.path.join(windows_font_dir, candidate)
            if os.path.exists(path):
                return path
                
    # Fallbacks for standard platform-independent locations or names
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
            
    return ""

def load_matching_font(bbox_height: int, font_family: str = "Sans-Serif", size_multiplier: float = 0.85, text: str = "") -> tuple:
    """
    Loads a PIL ImageFont matching the requested family (either standard or dynamic Google Font)
    and scaled to the bounding box height. Auto-detects Hindi/Devanagari characters and overrides
    fallbacks with appropriate Devanagari-supporting fonts.
    
    Parameters:
    - bbox_height: Height of the bounding box in pixels.
    - font_family: Font style family or custom Google Font name.
    - size_multiplier: Multiplier to adjust font size relative to bounding box height.
    - text: The replacement text content to analyze for character-level font requirements.
    
    Returns:
    A tuple (font_object, font_size)
    """
    font_size = max(8, int(bbox_height * size_multiplier))
    
    # 1. Detect if the text contains Devanagari (Hindi) characters
    is_hindi = False
    for char in text:
        if '\u0900' <= char <= '\u097F':
            is_hindi = True
            break
            
    # 2. Intercept and auto-resolve Hindi fonts
    if is_hindi:
        # If standard family is chosen, automatically upgrade to Google Noto Sans Devanagari
        if font_family in ["Sans-Serif", "Serif", "Monospace"]:
            font_family = "Noto Sans Devanagari"
            
    font_path = ""
    
    # 3. Check if a custom Google Font (or auto-resolved Hindi font) is requested
    if font_family not in ["Sans-Serif", "Serif", "Monospace"]:
        font_path = download_google_font(font_family)
        
    # 4. If it is standard family or download failed, resolve system fallback
    if not font_path:
        if is_hindi:
            # Fallback to standard Windows Devanagari system fonts
            windows_font_dir = "C:\\Windows\\Fonts"
            hindi_candidates = ["mangal.ttf", "utsah.ttf", "aparaj.ttf", "kokila.ttf"]
            for candidate in hindi_candidates:
                path = os.path.join(windows_font_dir, candidate)
                if os.path.exists(path):
                    font_path = path
                    break
        
        # If still no path resolved, fallback to standard system families
        if not font_path:
            standard_family = "Sans-Serif"
            if font_family in ["Serif", "Monospace"]:
                standard_family = font_family
            font_path = get_system_font_path(standard_family)
            
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
            return font, font_size
        except Exception as e:
            print(f"Failed to load TrueType font {font_path}: {e}")
            
    # Ultimate fallback to default PIL font
    try:
        font = ImageFont.load_default()
        return font, 12
    except Exception:
        return None, 0
