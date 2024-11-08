import base64
import os
import requests
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates as im_coordinates
import cv2
import numpy as np
from dotenv import load_dotenv
st.set_page_config(layout='wide')
load_dotenv()
def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

set_background('./bg.png')

api_endpoint = os.getenv("api_endpoint_mobilbit")

col01, col02 = st.columns(2)

# file uploader
file = col02.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Instructions for users
col02.write("**Click on the image where you want the background to be removed.**")

# read image
if file is not None:
    image = Image.open(file).convert('RGB')
    image = image.resize((880, int(image.height * 880 / image.width)))

    # create buttons
    col1, col2 = col02.columns(2)

    # visualize image
    placeholder0 = col02.empty()
    with placeholder0:
        value = im_coordinates(image)
        if value is not None:
            print(value)

    if col1.button('Original', use_container_width=True):
        placeholder0.empty()
        placeholder1 = col02.empty()
        with placeholder1:
            col02.image(image, use_column_width=True)

    if col2.button('Remove background', use_container_width=True):
        # call api
        placeholder0.empty()
        placeholder2 = col02.empty()

        filename = '{}_{}_{}.png'.format(file.name, value['x'], value['y'])

        if os.path.exists(filename):
            result_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        else:
            _, image_bytes = cv2.imencode('.png', np.asarray(image))
            image_bytes = image_bytes.tobytes()
            image_bytes_encoded_base64 = base64.b64encode(image_bytes).decode('utf-8')

            api_data = {"data": [image_bytes_encoded_base64, value['x'], value['y']]}
            response = requests.post(api_endpoint, json=api_data)

            result_image = response.json()['data']
            result_image_bytes = base64.b64decode(result_image)
            result_image = cv2.imdecode(np.frombuffer(result_image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            cv2.imwrite(filename, result_image)

        with placeholder2:
            col02.image(result_image, use_column_width=True)

        # Create download button
        with open(filename, "rb") as file:
            btn = col02.download_button(
                label="Download Image",
                data=file,
                file_name=filename,
                mime="image/png"
            )
