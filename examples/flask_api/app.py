# Import keras libraries
import keras
# Some keras models need unique preporecess functions
from keras.preprocessing import image
from keras.applications.xception import preprocess_input,decode_predictions
from keras.utils import multi_gpu_model

# Import basic libraries
import os
import time
import numpy as np
import tensorflow as tf
import imghdr
import cv2
from argparse import ArgumentParser

# Import flask libraries
import hashlib
import json
import requests
from urllib.parse import urlparse
from uuid import uuid4
from flask import Flask, jsonify, request, send_file, make_response, send_from_directory

# Directory of the project
ROOT_DIR = os.getcwd()
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

##################################################

def load_model():
    # Load the pre-trained Keras model
    global model
    model = keras.applications.xception.Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    # This is key : save the graph after loading the model
    global graph
    graph = tf.get_default_graph()

# Instantiate the app
app = Flask(__name__)

@app.route('/healthCheck', methods=['GET'])
def health_check():
    if request.method == 'GET':
        response = {
                'message': "OK."
            }
        return jsonify(response), 200

@app.route('/postImage', methods=['POST'])
def post_image():
    # Validate images extension
    image_type_ok_list = ['jpeg','png','gif','bmp']
    if 'file' not in request.files:
        response = {
            'message': "No file uploaded within the POST body."
        }
        return jsonify(response), 400

    # Response data
    data = {"success": False}

    # Run keras model
    uploaded_file = request.files['file']
    full_filename = os.path.join(IMAGE_DIR, uploaded_file.filename)
    uploaded_file.save(full_filename)

    img = image.load_img(full_filename, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    with graph.as_default():
	    preds = model.predict(x)

    results = decode_predictions(preds)

    # Return response
    if results:
        data["predictions"] = []
        for (imagenetID, label, prob) in results[0]:
            r = {"label": label, "probability": float(prob)}
            data["predictions"].append(r)
        data["success"] = True
        return jsonify(data), 200
    else:
        response_msg = "error"
        response = {
            'message': response_msg
        }
        return jsonify(response), 200

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=80, type=int, help='port to listen on')
    args = parser.parse_args()
    port = args.port

    load_model()
    app.run(host='0.0.0.0', port=port)
