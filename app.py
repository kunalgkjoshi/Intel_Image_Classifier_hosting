import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np

model = tf.keras.models.load_model(r'Intel_Image_Classifier_UI.h5')

st.write("""
         # Intel Image Classifier
         """
         )

st.write("This is a simple image classification web app to predict Building, Forest, Mountain, Street, Glacier and Sea.")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


def import_and_predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(224, 224),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is a Building!")
    elif np.argmax(prediction) == 1:
        st.write("It is a Forest!")
    elif np.argmax(prediction) == 2:
        st.write("It is a Glacier!")
    elif np.argmax(prediction) == 3:
        st.write("It is a Mountain!")
    elif np.argmax(prediction) == 4:
        st.write("It is a Sea!")
    else:
        st.write("It is a Street!")
    
