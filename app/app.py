
import streamlit as st
from streamlit_option_menu import option_menu
from pathlib import Path # <-- 1. IMPORT PATHLIB

# --- Core Tab Imports ---
from retina_tab import retina_tab
from dash_board_tab import dashboard_tab
from gradcam_tab import gradcam_tab
from image_segmentation_tab import image_segmentation_tab
from feature_extraction_tab import feature_extraction_tab
from augmentation_playground_tab import augmentation_playground_tab
from visual_guide_tab import visual_guide_tab

# --- Page Configuration ---
st.set_page_config(
    page_title="RetinoScan | AI Eye Diagnostics",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Path Settings (This is the new, robust part) ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "style.css"


# --- Custom CSS for a web-like feel ---
def load_css(file_name):
    """Function to load and inject a local CSS file."""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --- Load CSS using the new path ---
load_css(css_file) # <-- 2. CALL WITH THE FULL PATH


# --- Tab Management & App Logic ---
TABS = {
    # Title : (function, icon)
    "Visual Guide": (visual_guide_tab, "book"),
    "Retina Prediction": (retina_tab, "camera"),
    "Dashboard": (dashboard_tab, "bar-chart-line"),
    "Grad-CAM Heatmap": (gradcam_tab, "fire"),
    "Image Segmentation": (image_segmentation_tab, "palette"),
    "Feature Extraction": (feature_extraction_tab, "search-heart"),
    
} #"Augmentation": (augmentation_playground_tab, "images"),

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <img src="https://i.imgur.com/gJ52a26.png" width="60">
            <h1 style="color: #e0e0e0; margin-left: 10px;">RetinoScan</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    selected_tab_title = option_menu(
        menu_title=None,  # Hides the default menu title
        options=list(TABS.keys()),
        icons=[TABS[t][1] for t in TABS], # Gets icons from the dictionary
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
    )

    st.markdown("---")
    st.info("RetinoScan v1.1 | © 2025")


# --- Main Content Area ---

# Create a container for the main content to apply card styling
with st.container():
    # Retrieve the function and icon for the selected tab
    selected_tab_function, selected_tab_icon = TABS[selected_tab_title]

    # Display the selected tab's content by calling its function
    selected_tab_function()


# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <footer>
        Built with ❤️ using Streamlit | For Educational & Research Purposes Only
    </footer>
    """,
    unsafe_allow_html=True
)