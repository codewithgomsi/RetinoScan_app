# app/app.py
'''
import streamlit as st
from retina_tab import retina_tab
from gradcam_tab import gradcam_tab
from feature_extraction_tab import feature_extraction_tab
from dashboard_tab import dashboard_tab

st.set_page_config(page_title="RetinoScan", layout="wide")

st.sidebar.title("RetinoScan Navigation")
tab = st.sidebar.radio("Choose a Tab", [
    "Retina Image Prediction",
    "Grad-CAM Heatmap",
    "Feature Extraction & Research",
    "Metadata Dashboard"
])

if tab == "Retina Image Prediction":
    retina_tab()
elif tab == "Grad-CAM Heatmap":
    gradcam_tab()
elif tab == "Feature Extraction & Research":
    feature_extraction_tab()
elif tab == "Metadata Dashboard":
    dashboard_tab() '''
'''
# app/app.py
# ye functional h but simple so badme isse idea lena problem ke time 

import streamlit as st

# Core tabs
from retina_tab import retina_tab
#from odirk_tab import odirk_tab
#import dash_board_tab
from dash_board_tab import dashboard_tab
from gradcam_tab import gradcam_tab
from image_segmentation_tab import image_segmentation_tab
from feature_extraction_tab import feature_extraction_tab
from augmentation_playground_tab import augmentation_playground_tab
from visual_guide_tab import visual_guide_tab

# Optional tabs
#import future_condition_tab
#import confusion_matrix_tab

st.set_page_config(page_title="RetinoScan", layout="wide")

st.sidebar.title("RetinoScan Navigation")
tab = st.sidebar.selectbox("Choose a Tab", ["Visual Guide",
    "Retina Image Prediction",
    "ODIRK Disease Prediction",
    "Metadata Dashboard",
    "Grad-CAM Heatmap",
    "Feature Extraction & Research",
    "Image Segmentation","Data Augmentation Playground"
    "Future Retina Condition (AI)",
    "Confusion Matrix Viewer"
])

if tab == "Visual Guide":
    visual_guide_tab()
elif tab == "Retina Image Prediction":
    retina_tab()
elif tab == "Grad-CAM Heatmap":
    gradcam_tab()
elif tab == "Metadata Dashboard":
    dashboard_tab()
elif tab == "ODIRK Disease Prediction":
    odirk_tab.show()
elif tab == "Feature Extraction & Research":
    feature_extraction_tab()
elif tab == "Image Segmentation":
    image_segmentation_tab()
elif tab == "Data Augmentation Playground": 
    augmentation_playground_tab()
elif tab == "Future Retina Condition (AI)":
    future_condition_tab.show()
elif tab == "Confusion Matrix Viewer":
    confusion_matrix_tab.show()'''