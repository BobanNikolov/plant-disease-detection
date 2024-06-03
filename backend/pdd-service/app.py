from flask import Flask, request, jsonify
from flask_migrate import Migrate
import numpy as np
import tensorflow as tf
from keras.api.preprocessing.image import img_to_array, load_img
from flask_cors import CORS
from dotenv import load_dotenv
import os

from database import db
from models import Measurement

app = Flask(__name__)
load_dotenv()
model = tf.keras.models.load_model('model/weights_22epochs_final.weights.h5')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
migrate = Migrate(app, db)
cors = CORS(app)

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


def preprocess_image(image_path, target_size=(225, 225)):
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    return x


@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    image_path = './uploaded_image.jpg'
    image.save(image_path)

    x = preprocess_image(image_path)

    predictions = model.predict(x)

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
