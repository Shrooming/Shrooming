
import streamlit as st
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import os
import joblib
from skimage.feature import hog
from sympy import sympify
import cv2
from streamlit_drawable_canvas import st_canvas
import altair as alt
import pandas as pd

img_size = 45

st.title("Professor HOG!")
st.write("Give professor HOG a math question.")
st.image("proffessorhog.jpg")

uploaded_file = st.file_uploader("What's on your mind young one? Oinnk!", type=["jpg", "jpeg", "png"])

rf_hog = joblib.load("models/et_final_hog.pkl")
index_to_label = joblib.load("models/index_to_label.pkl")

classes = rf_hog.classes_
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
        if w < 2 or h < 2:
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
    predictions = []

    for s in symbols:
        label, proba = predict_symbol_rf_hog(s)
        predictions.append(label)

        # sortera top 5klasser
        top_idx = np.argsort(proba)[::-1][:5]
        df_top = pd.DataFrame({
            "Class": [index_to_label[i] for i in top_idx],
            "Confidence": [proba[i] for i in top_idx]
        })

        # checkbox för confidence
        if st.checkbox(f"Show Top-5 predictions for `{label}`", key=f"conf_{label}_{len(predictions)}"):
            chart = (
                alt.Chart(df_top)
                .mark_bar(color="blue")
                .encode(
                    x=alt.X("Confidence:Q", scale=alt.Scale(domain=[0,1])),
                    y=alt.Y("Class:N", sort="-x"),
                    tooltip=["Class", "Confidence"]
                )
                .properties(title=f"Top-5 predictions for `{label}`")
            )
            st.altair_chart(chart, use_container_width=True)

    return "".join(predictions)

def solving(expr_str):
    expr_str = expr_str.replace("times", "*")
    expr_str = expr_str.replace("div", "/")
    expr_str = expr_str.replace("sqrt", "**0.5")

    try:
        expr = sympify(expr_str)
        return expr.evalf()
    except Exception as e:
        return f"Error: {e}"

def predict_symbol_rf_hog(img):
    img = img.squeeze()
    hog_feat = hog(
        img,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    ).reshape(1, -1)

    proba = rf_hog.predict_proba(hog_feat)[0]  # sannolikheter
    pred_idx = np.argmax(proba)
    return index_to_label[pred_idx], proba




if uploaded_file is not None:
    math_image = Image.open(uploaded_file)
    st.image(math_image, use_container_width=True)

    expr_str = classify_expression(math_image)
    st.write(f"Professor HOG reads: `{expr_str}`")

    result = solving(expr_str)
    st.write(f"Answer: **{result}**")

    math_image = load_img(uploaded_file, color_mode="grayscale",
                          target_size=(img_size, img_size))
    img = img_to_array(math_image) / 255.0

st.write("Draw your math problem for the HOG")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",  # White background
    stroke_width=2,
    stroke_color="black",
    background_color="white",
    update_streamlit=True,
    height=200,
    width=400,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    drawn_img = Image.fromarray((canvas_result.image_data).astype("uint8")) # Convert canas to PIL img


    expr_str = classify_expression(drawn_img) # Classify
    st.write(f"Professor HOG sees: `{expr_str}`") #

    result = solving(expr_str)
    st.write(f"Professor HOGs result: **{result}**")