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
            if value is None:
                st.warning("Please click on the image first to select a point.")
            else:
                placeholder0.empty()
                placeholder2 = col02.empty()
                
                # Generate unique filename for cached results
                filename = '{}_{}_{}.png'.format(file.name, value['x'], value['y'])
                
                # Check if result already exists
                if os.path.exists(filename):
                    result_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                else:
                    # Convert PIL Image to numpy array
                    img_array = np.array(image)
                    
                    # Convert numpy array to bytes using OpenCV
                    _, img_encoded = cv2.imencode('.png', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                    image_bytes_encoded_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
                    
                    # Call API
                    api_data = {"data": [image_bytes_encoded_base64, value['x'], value['y']]}
                    response = requests.post(api_endpoint, json=api_data)
                    
                    # Process API response
                    result_image_base64 = response.json()['data']
                    result_image_bytes = base64.b64decode(result_image_base64)
                    result_image_array = np.frombuffer(result_image_bytes, dtype=np.uint8)
                    result_image = cv2.imdecode(result_image_array, cv2.IMREAD_UNCHANGED)
                    
                    # Save result
                    cv2.imwrite(filename, result_image)
                
                # Display result
                with placeholder2:
                    # Convert BGR to RGB for display
                    if len(result_image.shape) == 3 and result_image.shape[2] == 3:
                        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
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
        st.error("Error details:", exc_info=True)  # This will show more detailed error information