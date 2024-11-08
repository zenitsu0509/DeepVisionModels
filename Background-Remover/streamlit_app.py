import base64
import os
import requests
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates as im_coordinates
import numpy as np
from dotenv import load_dotenv
import cv2

# Page configuration
st.set_page_config(layout='wide')

# Load environment variables
load_dotenv()

# Get API endpoint from secrets
api_endpoint = st.secrets["api_endpoint_mobilbit"]

# Create layout
col01, col02 = st.columns(2)

# File uploader
file = col02.file_uploader('Upload an image', type=['jpeg', 'jpg', 'png'])

# Instructions for users
col02.write("**Click on the image where you want the background to be removed.**")

# Main application logic
if file is not None:
    try:
        # Read and resize image
        image = Image.open(file).convert('RGB')
        image = image.resize((880, int(image.height * 880 / image.width)))
        
        # Create button columns
        col1, col2 = col02.columns(2)
        
        # Image display placeholder
        placeholder0 = col02.empty()
        with placeholder0:
            value = im_coordinates(image)
            if value is not None:
                print(value)
        
        # Original image button
        if col1.button('Original', use_container_width=True):
            placeholder0.empty()
            placeholder1 = col02.empty()
            with placeholder1:
                col02.image(image, use_column_width=True)
        
        # Remove background button
        if col2.button('Remove background', use_container_width=True):
            placeholder0.empty()
            placeholder2 = col02.empty()
            
            # Generate unique filename for cached results
            filename = '{}_{}_{}.png'.format(file.name, value['x'], value['y'])
            
            # Check if result already exists
            if os.path.exists(filename):
                # Use PIL instead of cv2 for reading
                result_image = Image.open(filename)
            else:
                # Convert PIL Image to numpy array
                img_array = np.array(image)
                
                # Encode image using PIL and convert to base64
                pil_image = Image.fromarray(img_array)
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                image_bytes_encoded_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
                
                # Call API
                api_data = {"data": [image_bytes_encoded_base64, value['x'], value['y']]}
                response = requests.post(api_endpoint, json=api_data)
                
                # Process API response
                result_image_base64 = response.json()['data']
                result_image_bytes = base64.b64decode(result_image_base64)
                
                # Save and load with PIL
                with open(filename, 'wb') as f:
                    f.write(result_image_bytes)
                result_image = Image.open(filename)
            
            # Display result
            with placeholder2:
                col02.image(result_image, use_column_width=True)
            
            # Create download button
            with open(filename, "rb") as f:
                btn = col02.download_button(
                    label="Download Image",
                    data=f,
                    file_name=filename,
                    mime="image/png"
                )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")