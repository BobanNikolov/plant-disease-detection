import logging
import os

import joblib
import numpy as np
import tensorflow as tf
from PIL import Image
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_migrate import Migrate

from database import db
from models import Measurement

app = Flask(__name__)
load_dotenv()
classification_model = tf.keras.models.load_model('model/NewModel_15epochs_extended.weights.h5')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db.init_app(app)
migrate = Migrate(app, db)
cors = CORS(app)

# Compile the classification model
INIT_LR = 0.001
decay_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=INIT_LR,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True
)
opt = tf.keras.optimizers.Adam(learning_rate=decay_schedule)
classification_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Trained One-Class SVM model for detecting outliers
one_class_svm = joblib.load('model/one_class_svm_model_Optuna_100_newModel.pkl')

tomato_diseases_dict = {
    'Tomato_Bacterial_spot': 'Бактериска дамкавост',
    'Tomato_Early_blight': 'Црна дамкавост',
    'Tomato_Healthy': 'Здрав',
    'Tomato_Late_blight': 'Пламеница',
    'Tomato_Leaf_Mold': 'Плеснатост на листови',
    'Tomato_Mosaic_virus': 'Мозаичен вирус',
    'Tomato_Septoria_leaf_spot': 'Сива лисна дамкавост',
    'Tomato_Spider_mites': 'Пајакови грини',
    'Tomato_Target_Spot': 'Црна пегавост',
    'Tomato_Yellow_Leaf_Curl_Virus': 'Жолто лисно завивање'
}


def preprocess_image(image_path, target_size=(256, 256)):
    try:
        img = Image.open(image_path)
        img = img.resize(target_size)
        img = np.array(img).astype('float32') / 255.0  # Normalize pixel values to [0, 1]
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return None


# Check if an image is an outlier using One-Class SVM
def is_outlier_ocsvm(image_path, model, one_class_svm):
    img = preprocess_image(image_path)
    if img is None:
        return True  # Treat as outlier if feature extraction fails
    features = model.predict(img)
    prediction = one_class_svm.predict(features)
    return prediction[0] == -1


def classify_image(image_path, model):
    img = preprocess_image(image_path)
    if img is None:
        return "Unknown", 0.0
    predictions = model.predict(img)
    predicted_class_idx = np.argmax(predictions)
    predicted_probability = np.max(predictions)

    # Labels
    labels = {
        0: 'Tomato_Bacterial_spot', 1: 'Tomato_Early_blight', 2: 'Tomato_Healthy',
        3: 'Tomato_Late_blight', 4: 'Tomato_Leaf_Mold', 5: 'Tomato_Mosaic_virus',
        6: 'Tomato_Septoria_leaf_spot', 7: 'Tomato_Spider_mites', 8: 'Tomato_Target_Spot',
        9: 'Tomato_Yellow_Leaf_Curl_Virus'
    }

    predicted_label = labels[predicted_class_idx]

    return predicted_label, predicted_probability


def print_classification_result(label, note=None, valid_classification=False):
    if valid_classification:
        print("\n--- Valid Classification ---")
    else:
        print("\n--- Classification Result ---")
    print(f"Predicted Disease: {label}")
    if note:
        print(f"Note: {note}")
    print("------------------------------\n")


# Classify or detect outlier
def classify_or_detect_outlier(image_path, classification_model, one_class_svm):
    if is_outlier_ocsvm(image_path, classification_model, one_class_svm):
        note = "Unknown object (outlier detected)"
        print_classification_result("Unknown", note=note)
        return "Unknown"
    else:
        label, probability = classify_image(image_path, classification_model)
        if probability > 0.99:
            note = "This prediction might be an outlier due to high confidence."
            print_classification_result(label, note)
            return "Unknown"
        elif probability < 0.70:
            note = "This prediction might be an outlier due to low probability."
            print_classification_result(label, note)
            return "Unknown"
        else:
            print_classification_result(label, valid_classification=True)
            return label


@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    image = request.files['image']
    image_path = './uploaded_image.jpg'
    image.save(image_path)

    # Classify or detect outlier
    predictions = classify_or_detect_outlier(image_path, classification_model, one_class_svm)

    if predictions == "Unknown":
        return jsonify({'prediction': predictions, 'prediction_translated': 'Непозната вредност'})

    labels = {'Tomato_Bacterial_spot': 0, 'Tomato_Early_blight': 1, 'Tomato_Healthy': 2, 'Tomato_Late_blight': 3,
              'Tomato_Leaf_Mold': 4, 'Tomato_Mosaic_virus': 5, 'Tomato_Septoria_leaf_spot': 6, 'Tomato_Spider_mites': 7,
              'Tomato_Target_Spot': 8, 'Tomato_Yellow_Leaf_Curl_Virus': 9}

    labels = {v: k for k, v in labels.items()}
    predicted_label = labels[np.argmax(predictions)]

    return jsonify({'prediction': predicted_label, 'prediction_translated': tomato_diseases_dict[predicted_label]})


@app.route('/measurement', methods=['POST'])
def measurement_save():
    data = request.get_json()
    lon = data.get('lon')
    lat = data.get('lat')
    time_of_measurement = data.get('time_of_measurement')
    predicted_result = data.get('predicted_result')
    measurement = Measurement(None, lon, lat, time_of_measurement, predicted_result)
    db.session.add(measurement)
    db.session.commit()
    return jsonify(measurement.to_dict())


@app.before_request
def app_context():
    db.create_all()


if __name__ == '__main__':
    app.run(debug=os.getenv('DEBUG'))
