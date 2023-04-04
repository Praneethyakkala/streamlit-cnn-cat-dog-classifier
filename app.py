import streamlit as st
from keras_preprocessing import image
from keras.models import load_model
import numpy as np
import io
from PIL import Image
#import tensorflow as tf


"""Load model once at running time for all the predictions"""
print('[INFO] : Model loading ................')
global model
model = load_model('cat_dog_classifier.h5')
#global graph
#graph = tf.get_default_graph()
print('[INFO] : Model loaded')

st.title('What is this image? :cat: :dog:')

global bytes_data
uploaded_file = st.file_uploader("Upload a file to classify", label_visibility = "collapsed")
if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    img = Image.open(StringIO(bytes_data))
    global data
    data = Image.resize(img, (128, 128, 3))
    
def predict():
        #data = image.load_img(bytes_data, target_size=(128, 128, 3))
        # (150,150,3) ==> (1,150,150,3)
        data = np.expand_dims(data, axis=0)

        # Scaling
        data = data.astype('float') / 255

        # Prediction
        #with graph.as_default():
        result = model.predict(data)

        pred_prob = result.item()

        if pred_prob > .5:
            label = 'Dog'
            accuracy = round(pred_prob * 100, 2)
        else:
            label = 'Cat'
            accuracy = round((1 - pred_prob) * 100, 2)
        
        st.success('This is a ' + label + ' predicted with confidence +' + str(accuracy))


trigger = st.button('Predict', on_click=predict)
               
