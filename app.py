import os
import base64
import streamlit as st

from utils import load_model, infer_uploaded_image
from pathlib import Path

from config import model_config 

st.cache(allow_output_mutation=True)

# Page layout
st.set_page_config(
    page_title="YOLOv10 Detection Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page title
col = st.container()   
with col:
    st.title(':sparkles: :blue[YOLOv10] Detection Demo')
    st.text('Model: Pre-trained YOLOv10n')

# Sidebar
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection"]
)

model_type = None
if task_type == "Detection":
    model_type = st.sidebar.selectbox(
        "Select Model",
        model_config.DETECTION_MODEL_LIST
    )
else:
    st.error("Please select a task type")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100

model_path = ""
if model_type == "YOLOv10n":
    model_path = Path(model_config.DETECTION_MODEL_DIR, str(model_type))
    
else:
    st.error("Please select a model type")
    
# Load pretrained DL model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Error loading model. Please check the specified model path: {model_path}")
    
# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    model_config.SOURCES_LIST
)

source_img = None
if source_selectbox == model_config.SOURCES_LIST[0]: # Image
    infer_uploaded_image(confidence, model)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")