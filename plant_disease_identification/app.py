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

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = '/home/dit/DiT/GitHub/Pylingo/Jupyters/DS/Keras/saved_model/plant/tomato_inception_v3.h5'

# Load your trained model
model = load_model(MODEL_PATH, custom_objects=None, compile=True)

#%%
# You can also use pretrained model from Keras
# Check https://keras.io/applications/

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    from keras.preprocessing import image

    IMAGE_SIZE = [224, 224]
    labels = ['Bacterial_spot','Early_blight','Late_blight','Leaf_Mold','Septoria_leaf_spot',
              'Spider_mites Two-spotted_spider_mite','Target_Spot','Tomato_Yellow_Leaf_Curl_Virus',
              'Tomato_mosaic_virus','healthy']

    img = image.load_img(img_path, target_size=IMAGE_SIZE)

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.array(x).astype('float32')/255
    x = np.reshape(x, IMAGE_SIZE + [3])
    x = np.expand_dims(x, axis=0)

    result = model.predict(x)

    what_class = np.argmax(result, axis=-1)
    scale = '{:.2f}'.format(round(result.max(), 2))
    if what_class == 9:
        return (f'Tomato is classified HEALTHY with scale of {result.max()}')

    else:
        return (f'Infected with {labels[what_class[0]].upper()} with confident scale of {scale}')


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
        f.save(file_path)

        # Make prediction
        r = model_predict(file_path, model)
        return r


        # # Process your result for human
        # # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred[0][0])               # Convert to string
        # return pred
    return None


if __name__ == '__main__':
    app.run(debug=True)

