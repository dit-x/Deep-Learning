#!/home/dit/PACKAGES/.ds_env/bin/python3

#%%
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = '/home/dit/DiT/GitHub/Pylingo/Jupyters/DS/Keras/saved_model/cats_and_dogs_small_pretrained_2.h5'

# Load your trained model
model = load_model(MODEL_PATH, custom_objects=None, compile=True)

#%%
# You can also use pretrained model from Keras
# Check https://keras.io/applications/

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):

    img = image.load_img(img_path, target_size=(150, 150))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.array(x).astype('float32')/255
    x = np.reshape(x, (150, 150, 3))
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))

        # SAVE THE PICTURE
        # f.save(file_path)

        # Make prediction
        pred = model_predict(file_path, model)

        if pred > 0.5 :
            result = ('Dog - {:0.2f}'.format(pred[0][0]))
            return result
        
        else:
            result = ('Cat - {:0.2f}'.format(pred[0][0]))
            return result


        # # Process your result for human
        # # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred[0][0])               # Convert to string
        # return pred
    return None


if __name__ == '__main__':
    app.run(debug=True)

