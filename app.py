
import streamlit as st
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import os
import joblib
from skimage.feature import hog
from sympy import sympify
import cv2

img_size = 45

st.title("Professor HOG!")
st.write("Give professor HOG a math question.")
st.image("proffessorhog.jpg")

uploaded_file = st.file_uploader("What's on your mind young one? Oinnk!", type=["jpg", "jpeg", "png"])

rf_hog = joblib.load("models/rf_hog.pkl")
index_to_label = joblib.load("models/index_to_label.pkl")

def predict_symbol_rf_hog(img):
    img = img.squeeze()
    hog_feat = hog(
        img,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    ).reshape(1, -1)

    pred_idx = rf_hog.predict(hog_feat)[0]
    return index_to_label[pred_idx]

def segment_expression(img_pil, img_size=45):
    img = np.array(img_pil.convert("L"))
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    symbol_imgs, bounding_boxes = [], []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 5:
            continue

        roi = thresh[y:y+h, x:x+w]

        side = max(w, h)
        square = np.zeros((side, side), dtype=np.uint8)
        square[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w] = roi

        roi_resized = cv2.resize(square, (img_size, img_size), interpolation=cv2.INTER_AREA)
        roi_resized = roi_resized.astype("float32") / 255.0
        roi_resized = np.expand_dims(roi_resized, axis=-1)

        symbol_imgs.append(roi_resized)
        bounding_boxes.append((x, y, w, h))

    # Sort left-to-right
    sorted_symbols = [s for _, s in sorted(zip(bounding_boxes, symbol_imgs), key=lambda b: b[0][0])]

    return sorted_symbols

def classify_expression(img_pil):
    symbols = segment_expression(img_pil, img_size=45)
    predicted_chars = [predict_symbol_rf_hog(s) for s in symbols]
    return "".join(predicted_chars)

def classify_expression(img_pil):
    symbols = segment_expression(img_pil, img_size=45)
    predicted_chars = [predict_symbol_rf_hog(s) for s in symbols]
    return "".join(predicted_chars)

def solving(expr_str):
    expr_str = expr_str.replace("times", "*")
    expr_str = expr_str.replace("div", "/")
    expr_str = expr_str.replace("sqrt", "**0.5")

    try:
        expr = sympify(expr_str)
        return expr.evalf()
    except Exception as e:
        return f"Error: {e}"


st.write("very gud")

if uploaded_file is not None:
    math_image = Image.open(uploaded_file)
    st.image(math_image, use_column_width=True)

    expr_str = classify_expression(math_image)
    st.write(f"Professor HOG reads: `{expr_str}`")

    result = solving(expr_str)
    st.write(f"Answer: **{result}**")

    math_image = load_img(uploaded_file, color_mode="grayscale",
                          target_size=(img_size, img_size))
    img = img_to_array(math_image) / 255.0
