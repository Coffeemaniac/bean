import numpy as np
import os
from PIL import Image, ImageOps

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

import streamlit as st

import base64
from io import BytesIO

def run_model(image):
    img_data = image.read()

    test_image = load_img(img_data, target_size = (150, 150)) # load image 
    
    test_image = img_to_array(test_image)/255 # convert image to np array and normalize

    test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
    
    result = model.predict(test_image).round(3) # predict diseased palnt or not
    
    pred = np.argmax(result) # get the index of max value

    if pred == 0:
        st.write("The chosen image is of a *healthy plant*.")
    elif pred == 1:
        st.write("The chosen image is of a *diseased plant*. More details about cotton plant diseases can be found here.")
    elif pred == 2:
        st.write("The chosen image is of a *healthy plant* but the leaves might be experiencing a burn.")
    else:
        st.write("The chosen image is of a *healthy plant* but the leaves might be experiencing a burn.")

def load_image(image):
    img = Image.open(image)
    return img

def main():
    # load the model
    model = load_model("model/v3_pred_cott_dis.h5")

    st.title("beanco.ml")

    menu = "Home"

    menu = st.sidebar.selectbox("Menu", ("Home", "Policy"))

    if menu == "Home":
        st.write("""
            Bean helps diagnose crop diseases, users are required to upload a picture of the leaf and it predicts whether the plant has a disease or not using artificial intelligence.
        """)
        st.write("-----------")
        
        uploaded_file = st.file_uploader(label="Upload a JPG file", type="JPG")

        if uploaded_file is not None:
            try:
                img = load_image(uploaded_file)
                st.sidebar.image(img, width=300)
                st.write(type(img))
            except Exception as e:
                print(e)
    if menu == "Policy":
        st.empty()
        st.write("""
            Here goes the policy
        """)

if __name__ == '__main__':
    main()
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)