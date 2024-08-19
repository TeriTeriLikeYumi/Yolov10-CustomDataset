from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import numpy as np

def load_model(model_path):
    model = YOLO(model_path) # Load the model
    return model

def _display_detected_frames(conf, model, st_frame, image):
    res = model.predict(image, conf = conf) # Predict the bounding boxes
    
    # Plot the detected objects on the video frame 
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected objects',
                   channels='BGR',
                   use_column_width=True)
    return res
    
def infer_uploaded_image(conf, model):
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=['jpg', 'jpeg', 'png','bmp','webp']
    )
    
    col1, col2 = st.columns(2)
    
    
    if source_img is not None:
        uploaded_image = Image.open(source_img)
        # Convert the PIL image to a OpenCV format
        uploaded_image = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
        
        with col1:
            # Adding image to page with caption
            st.image(
                image = source_img,
                caption = 'Uploaded Image',
                use_column_width = True
            )
            
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    st.frame = st.empty()
                    res = _display_detected_frames(conf, model, st.frame, uploaded_image)
        
                    with col2:
                        st.image(res[0].plot(), caption="Detected Image", use_column_width=True)
                
                    with st.expander("Detection Results"):
                        for box in res[0].boxes:
                            st.write(box.xywh)
                except Exception as ex:
                    st.write("No image is uploaded yet!")
                    st.write(ex)
        