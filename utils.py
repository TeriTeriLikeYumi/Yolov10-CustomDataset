import os
import cv2
import wget
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np

# def download_model():
#     url = 'https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt'
#     dest_path = 'model/yolov10/weights'
#     os.makedirs(dest_path, exist_ok=True)
#     wget.download(url, out=dest_path)

def load_model(model_path):
    model = YOLO(model_path) # Load the model
    return model

def infer_uploaded_image(conf, model):
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=['jpg', 'jpeg', 'png','bmp','webp']
    )
    
    col1, col2 = st.columns(2)
    
    
    if source_img is not None:
        uploaded_image = Image.open(source_img)
        uploaded_image = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
        with col1:
            # Adding image to page with caption
            st.image(
                image = uploaded_image,
                caption = 'Uploaded Image',
                use_column_width = True
            )
            
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    st.frame = st.empty()
                    res = model(uploaded_image, conf)
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:,:,::-1]
                    with col2:
                        st.image(res_plotted, caption="Detected Image", use_column_width=True)
                
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.xywh)
                except Exception as ex:
                    st.write("No image is uploaded yet!")
                    st.write(ex)
    else:
        st.info("Please upload an image to start the detection process")