# Author: Parth Mehta

import time
from PIL import Image
import streamlit as st

from model import detect

def main():
    # Header & Page Config.
    st.set_page_config(
        page_title="Parth - Detr",
        layout="centered")
    st.title("Object Detection using DETR:")

    # This will let you upload PNG, JPG & JPEG File
    buffer = st.file_uploader("Upload your Image here", type = ["jpg", "png", "jpeg"])

    if buffer:
        # Object Detecting
        with st.spinner('Wait for it...'):
            # Slider for changing confidence
            confidence = st.slider('Confidence Threshold', 0, 100, 90)

            # Calculating time for detection
            t1 = time.time()
            im = Image.open(buffer)
            im.save("saved_images/image.jpg")
            res_img = detect("saved_images/image.jpg", confidence)
            t2 = time.time()
        
        # Displaying the image
        st.image(res_img, use_column_width = True)
        
        # Printing Time
        st.write("\n")
        st.write("Time taken: ", t2-t1, "sec.")
            

if __name__ == '__main__':
	main()