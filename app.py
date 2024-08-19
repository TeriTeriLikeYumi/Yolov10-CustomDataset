import streamlit as st
from pathlib import Path

# Local modules
from config import model_config 
from components import footer
import utils

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

# Sidebar header
st.sidebar.header("Model Config")
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection"]
)

# Sidebar model selection
model_type = None
if task_type == "Detection":
    model_type = st.sidebar.selectbox(
        "Select Model",
        model_config.DETECTION_MODEL_LIST
    )
else:
    st.error("Please select a task type")

# Sidebar model confidence
confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100

# Get model path
@st.cache_resource
def get_model_path(model_type):
    model_path = ""
    if model_type == "YOLOv10n":
        model_path = Path(model_config.DETECTION_MODEL_DIR, str(model_type))
    else:
        st.error("Please select a model type")
    return model_path
    
# Load pretrained ML model
@st.cache_resource
def load_model(model_path):
    try:
        model = utils.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model. Please check the specified model path: {model_path}")
    return model

# Load model
model_path = get_model_path(model_type)
model = load_model(model_path)

# Image options
st.sidebar.header("Image Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    model_config.SOURCES_LIST
)

# Image Source
source_img = None
utils.infer_uploaded_image(confidence, model)

footer.footer()