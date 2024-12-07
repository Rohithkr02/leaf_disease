import streamlit as st
from ultralytics import YOLO
import requests
from PIL import Image
import io
import cv2
import numpy as np

# Function to check if the image URL is accessible
# def check_image_accessibility(image_url):
#     try:
#         response = requests.get(image_url)
#         if response.status_code == 200:
#             return True
#         elif response.status_code == 403:
#             st.error("403 Forbidden: You do not have permission to access this image.")
#             return False
#         else:
#             st.error(f"Error {response.status_code}: Unable to access the image.")
#             return False
#     except Exception as e:
#         st.error(f"An error occurred: {e}")
#         return False

# Load YOLO model
model = YOLO("D:/rohith/Duk/dlmini/best2.pt")  # Path to your trained YOLO model

# Streamlit UI for image upload or URL input
st.title("Leaf Disease Detection")

# Input options
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        img = Image.open(uploaded_image)

        # Convert uploaded image to OpenCV format
        # img_array = np.array(uploaded_image)
        # img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Perform prediction
        results = model.predict(img, imgsz=640, conf=0.25)
        for r in results:
            annotated_img = r.plot()
            st.image(annotated_img, caption="Annotated Image", use_column_width=True)
# else:
#     image_url = st.text_input("Enter Image URL:")
#     if image_url:
#         if check_image_accessibility(image_url):
#             st.write("Image is accessible. Performing prediction...")
#             # Download image from URL
#             img_data = requests.get(image_url).content
#             image = Image.open(io.BytesIO(img_data))
#             st.image(image, caption="Image from URL", use_column_width=True)
            
#             # Convert image to OpenCV format
#             img_array = np.array(image)
#             img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
#             # Perform prediction
#             results = model.predict(img_bgr, imgsz=640, conf=0.25)
#             for r in results:
#                 annotated_img = r.plot()
#                 st.image(annotated_img, caption="Annotated Image", use_column_width=True)

