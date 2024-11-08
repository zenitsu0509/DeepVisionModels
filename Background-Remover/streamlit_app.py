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

file = col1.file_uploader('Upload an image', type=['jpeg', 'jpg', 'png'])

col1.write("**Click on the image where you want the background to be removed.**")

if file is not None:
    try:
        image = Image.open(file).convert('RGB')
        image = image.resize((880, int(image.height * 880 / image.width)))

        placeholder0 = col1.empty()
        with placeholder0:
            value = im_coordinates(image)
            if value is not None:
                print(value)

        if col2.button('Original', use_container_width=True):
            placeholder0.empty()
            placeholder1 = col1.empty()
            with placeholder1:
                col1.image(image, use_column_width=True)

        if col2.button('Remove background', use_container_width=True):
            if value is None:
                st.warning("Please click on the image first to select a point.")
            else:
                placeholder0.empty()
                placeholder2 = col1.empty()

                filename = '{}_{}_{}.png'.format(file.name, value['x'], value['y'])

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
                    col1.image(result_image, use_column_width=True)

                with open(filename, "rb") as f:
                    btn = col1.download_button(
                        label="Download Image",
                        data=f,
                        file_name=filename,
                        mime="image/png"
                    )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Error details:", exc_info=True)