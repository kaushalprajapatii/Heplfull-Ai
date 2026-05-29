import os
import requests
import re

def clean_font_name(font_name: str) -> str:
    """Cleans up font family name and formats it properly (e.g. Poppins, Noto Sans Devanagari)."""
    # Strip whitespace and capitalize words properly
    parts = [part.strip().capitalize() for part in font_name.split(" ") if part.strip()]
    return " ".join(parts)

def download_google_font(font_family: str) -> str:
    """
    Downloads a font family from Google Fonts by leveraging the CSS API with an older
    User-Agent to fetch raw TrueType (.ttf) URLs, requesting the Devanagari and Latin subsets,
    and caches them locally.
    
    Returns:
    Absolute path to the cached TTF file, or empty string if it fails.
    """
    cleaned_name = clean_font_name(font_family)
    if not cleaned_name:
        return ""
        
    # Local fonts cache directory in backend/text_editor/fonts/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fonts_dir = os.path.join(base_dir, "fonts")
    os.makedirs(fonts_dir, exist_ok=True)
    
    # Cache filename prefix e.g. "poppins_regular.ttf" or "notosansdevanagari_regular.ttf"
    safe_prefix = cleaned_name.lower().replace(" ", "")
    cache_path = os.path.join(fonts_dir, f"{safe_prefix}_regular.ttf")
    
    # If already cached, verify that it contains full character support (over 35 KB)
    # Subsetted Latin-only fonts are typically under 25 KB. Devanagari subsets are >100 KB.
    if os.path.exists(cache_path):
        if os.path.getsize(cache_path) > 35000 or "devanagari" not in safe_prefix:
            return cache_path
        else:
            print(f"Cached font '{cleaned_name}' is subsetted Latin-only ({os.path.getsize(cache_path)} bytes). Upgrading to Devanagari subset...")
            
    # Fetch Google Fonts CSS API using an iOS 4 Safari User-Agent to force TTF formats
    # Request Devanagari and Latin subsets to ensure full multi-lingual rendering support
    url_name = cleaned_name.replace(" ", "+")
    css_url = f"https://fonts.googleapis.com/css?family={url_name}&subset=devanagari,latin,latin-ext"
    headers = {
        'User-Agent': 'Mozilla/5.0 (iPhone; U; CPU iPhone OS 4_2_1 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8C148 Safari/5653.8'
    }
    
    print(f"Searching Google Fonts for '{cleaned_name}' using: {css_url}")
    
    try:
        response = requests.get(css_url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Font '{cleaned_name}' not found on Google Fonts (CSS status code: {response.status_code})")
            return ""
            
        # Parse direct .ttf URL from the CSS body
        match = re.search(r'url\((https://[^\)]+\.ttf)\)', response.text)
        if not match:
            print(f"No direct TrueType (.ttf) URL found in Google Fonts CSS for '{cleaned_name}'.")
            return ""
            
        ttf_url = match.group(1)
        print(f"Downloading raw TTF file from gstatic: {ttf_url}")
        
        # Download the binary TTF content
        font_response = requests.get(ttf_url, timeout=10)
        if font_response.status_code != 200:
            print(f"Failed to download TTF binary from gstatic (HTTP status: {font_response.status_code})")
            return ""
            
        # Write to local cache directory
        with open(cache_path, "wb") as f:
            f.write(font_response.content)
            
        print(f"Successfully downloaded and cached Google Font: '{cleaned_name}' ({len(font_response.content)} bytes) -> {cache_path}")
        return cache_path
        
    except Exception as e:
        print(f"Error dynamically acquiring Google Font '{cleaned_name}': {e}")
        return ""
