import base64
import os
import requests
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates as im_coordinates
import numpy as np
from dotenv import load_dotenv
import cv2

st.set_page_config(layout='wide')

load_dotenv()

api_endpoint = st.secrets["api_endpoint_mobilbit"]

col1, col2 = st.columns(2)

# File uploader
file_uploader = col1.file_uploader('Upload an image', type=['jpeg', 'jpg', 'png'])


original_button = col1.button('Original')
remove_bg_button = col1.button('Remove background')

if file_uploader is not None:
    try:
        image = Image.open(file_uploader).convert('RGB')
        
        placeholder0 = col1.empty()
        value = None
        
        with placeholder0:
            value = im_coordinates(image)
            if value is not None:
                print(value)
                
        
        col1.image(image, use_column_width=False)

        
        if original_button:
            placeholder0.empty()
            col1.image(image, use_column_width=False)

        
        if remove_bg_button:
            if value is None:
                st.warning("Please click on the image first to select a point.")
            else:
                placeholder0.empty()
                placeholder2 = col1.empty()

                filename = f"{file_uploader.name}_{value['x']}_{value['y']}.png"

             
                if os.path.exists(filename):
                    result_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                else:
                    img_array = np.array(image)
                    _, img_encoded = cv2.imencode('.png', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                    image_bytes_encoded_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

                    api_data = {"data": [image_bytes_encoded_base64, value['x'], value['y']]}
                    response = requests.post(api_endpoint, json=api_data)

                    result_image_base64 = response.json()['data']
                    result_image_bytes = base64.b64decode(result_image_base64)
                    result_image_array = np.frombuffer(result_image_bytes, dtype=np.uint8)
                    result_image = cv2.imdecode(result_image_array, cv2.IMREAD_UNCHANGED)

                    
                    cv2.imwrite(filename, result_image)

                with placeholder2:
                    if len(result_image.shape) == 3 and result_image.shape[2] == 3:
                        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    col1.image(result_image, use_column_width=False)

                with open(filename, "rb") as f:
                    col1.download_button(
                        label="Download Image",
                        data=f,
                        file_name=filename,
                        mime="image/png"
                    )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
