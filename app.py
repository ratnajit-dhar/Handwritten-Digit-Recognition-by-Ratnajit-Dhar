import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("digit_recognition.h5")
    return model

model = load_model()

def preprocess_image(image):
    image = image.convert('L')
    image = image.resize((28,28))
    image = np.array(image)
    image = image/255.0
    image = np.expand_dims(image, axis=-1)
    return image

def predict(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(np.array([preprocessed_image]))
    return prediction

st.title("Handwritten Digit Recognition")
st.write("Made by Ratnajit Dhar")
st.header("Draw the digit(Try to draw it centrally)")

st.sidebar.title("Social Media Links")
st.sidebar.markdown("[Facebook](https://www.facebook.com/RatnajitDharPrantar/)")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/ratnajit-dhar/)")
st.sidebar.markdown("[Github](https://github.com/ratnajit-dhar)")
st.sidebar.title("Github Repo for this Project")
st.sidebar.markdown("[Handwritten-Digit-Recognition-by-Ratnajit-Dhar](https://github.com/ratnajit-dhar/ratnajit-dhar-Handwritten-Digit-Recognition.git)")

drawing_mode = "freedraw"

canvas = st_canvas(
    stroke_width = 15,
    stroke_color = '#FFFFFF',
    background_color = '#000000',
    height = 280,
    width = 280,
    key = "canvas",
)

if(st.button('Predict')):
    if canvas.image_data is not None:
        image_array = np.array(canvas.image_data)
        image = Image.fromarray(image_array)
        prediction = predict(image)
        pred = prediction.argmax(axis=1)
        st.write(f"The Digit might be:")
        st.header(f"{pred[0]}")
    else:
        st.write(f"Draw a Digit first")
