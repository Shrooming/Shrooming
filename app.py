
import streamlit as st
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import os

img_size = 45

st.title("Professor HOG!")
st.write("Give professor HOG a math question.")
st.image("proffessorhog.jpg")



uploaded_file = st.file_uploader("What's on your mind young one? Oinnk!", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    math_image = Image.open(uploaded_file)
    st.image(math_image, use_column_width=True)

    #Preproccesing
    math_image = load_img(uploaded_file, color_mode="grayscale",
    target_size=(img_size, img_size))
    img = img_to_array(math_image) / 255.0

st.write("very gud")

