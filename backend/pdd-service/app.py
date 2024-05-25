from flask import Flask, request, jsonify
from flask_migrate import Migrate
import numpy as np
import tensorflow as tf
from keras.api.preprocessing.image import img_to_array, load_img

from database import db
from models import Measurement

app = Flask(__name__)
model = tf.keras.models.load_model('model/weights_22epochs_final.weights.h5')
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://plant_disease:plant_disease@localhost:5433/plant_disease'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
migrate = Migrate(app, db)

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
    print(predictions[0])

    labels = {'Tomato_Bacterial_spot': 0, 'Tomato_Early_blight': 1, 'Tomato_Healthy': 2, 'Tomato_Late_blight': 3,
              'Tomato_Leaf_Mold': 4, 'Tomato_Mosaic_virus': 5, 'Tomato_Septoria_leaf_spot': 6, 'Tomato_Spider_mites': 7,
              'Tomato_Target_Spot': 8, 'Tomato_Yellow_Leaf_Curl_Virus': 9}
    labels = {v: k for k, v in labels.items()}
    predicted_label = labels[np.argmax(predictions)]

    return jsonify({'prediction': predicted_label})

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
    app.run(debug=True)
