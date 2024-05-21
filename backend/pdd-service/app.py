from flask import Flask, request, jsonify

import tensorflow as tf
from keras.api.models import load_model
from keras.api.preprocessing.image import img_to_array, load_img
import numpy as np

app = Flask(__name__)
model = load_model('../model/weights_22epochs_NewData_OldModel.weights.h5')


def preprocess_image(image_path, target_size=(225, 225)):
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    return x


@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    image = request.files['image']
    image_path = './uploaded_image.jpg'
    image.save(image_path)

    # Preprocess the image
    x = preprocess_image(image_path)

    # Make prediction
    predictions = model.predict(x)

    # Decode prediction (assuming labels are available)
    labels = {'Tomato_Bacterial_spot': 0, 'Tomato_Early_blight': 1, 'Tomato_Healthy': 2, 'Tomato_Late_blight': 3,
              'Tomato_Leaf_Mold': 4, 'Tomato_Mosaic_virus': 5, 'Tomato_Septoria_leaf_spot': 6, 'Tomato_Spider_mites': 7,
              'Tomato_Target_Spot': 8, 'Tomato_Yellow_Leaf_Curl_Virus': 9}
    labels = {v: k for k, v in labels.items()}
    predicted_label = labels[np.argmax(predictions)]

    # Return prediction
    return jsonify({'prediction': predicted_label})


if __name__ == '__main__':
    app.run(debug=True)
