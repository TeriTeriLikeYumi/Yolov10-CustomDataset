from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np

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
        image = Image.open(source_img)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        with col1:
            # Adding image to page with caption
            st.image(
                image = image,
                caption = 'Uploaded Image',
                use_column_width = True
            )
            
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    st.frame = st.empty()
                    image_array = np.array(image)
                    res = list(model.predict(image_array, conf))
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()
                    with col2:
                        st.image(image = res_plotted, 
                                 caption = "Detected Image", 
                                 use_column_width = True
                                 )
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.xywh)
                except Exception as ex:
                    st.write("No image is uploaded yet!")
                    st.write(ex)
    else:
        st.info("Please upload an image to start the detection process")