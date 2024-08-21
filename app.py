import streamlit as st
from pathlib import Path
import supervision as sv

# Local modules
import model_config 
from components import footer
import utils

# Setting Page layout
st.set_page_config(
    page_title="YOLOv10 Detection Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setting Page title
col = st.container()   
with col:
    st.title(':sparkles: :blue[YOLOv10] Detection Demo')
    st.text('Model: Pre-trained YOLOv10n')

# Sidebar  
st.sidebar.header("Model Config")
model_type = st.sidebar.selectbox(
    "Select Model",
    ["Detection", "Custom"]
)
if model_type == None:
    st.error("Please select a task type")
# Sidebar model confidence
confidence = float(st.sidebar.slider(
    "Select Model Confidence", 10, 100, 20)) / 100

# Get model path
@st.cache_resource
def get_model_path(model_type):
    if model_type == "Detection":
        model_path = Path(model_config.DETECTION_MODEL)
    elif model_type == "Fruits Detection":
        model_path = Path(model_config.CUSTOM_MODEL)
    else:
        st.error("Please select a model type")
    return model_path
    
# Load pretrained ML model
@st.cache_resource
def load_model(model_path):
    try:
        model = utils.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model. Please check the specified model path: {model_path}")
        st.error(e)

# Load model
model_path = get_model_path(model_type)
model = load_model(model_path)

# Image options
st.sidebar.header("Image Config")
source_selectbox = model_config.SOURCES_LIST

# Image Source
source_img = None
utils.infer_uploaded_image(confidence, model)

footer.footer()