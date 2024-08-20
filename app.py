import streamlit as st
from pathlib import Path
import supervision as sv

# Local modules
from config import model_config 
from components import footer
from utils import download_model, load_model, infer_uploaded_image

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
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection"]
)

# Sidebar model selection
model_type = None
if task_type == "Detection":
    model_type = model_config.DETECTION_MODEL
else:
    st.error("Please select a task type")

# Sidebar model confidence
confidence = float(st.sidebar.slider(
    "Select Model Confidence", 10, 100, 20)) / 100

# Get model path
@st.cache_resource
def get_model_path(model_type):
    model_path = ""
    if model_type == "yolov10n.pt":
        model_path = Path(model_config.MODEL_DIR, str(model_type))
    else:
        st.error("Please select a model type")
    return model_path
    
# Load pretrained ML model
@st.cache_resource
def load_model(model_path):
    try:
        model = load_model(model_path)
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
infer_uploaded_image(confidence, model)

footer.footer()