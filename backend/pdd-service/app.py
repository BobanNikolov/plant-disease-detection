from datetime import datetime

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from keras.api.preprocessing.image import img_to_array, load_img

app = Flask(__name__)
model = tf.keras.models.load_model('model\weights_22epochs_final.weights.h5')
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://plant_disease:plant_disease@localhost:5433/plant_disease'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

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
    lon = request.form.get('lon')
    lat = request.form.get('lat')
    time_of_measurement = request.form.get('time_of_measurement')
    predicted_result = request.form.get('predicted_result')
    measurement = Measurement(None, lon, lat, time_of_measurement, predicted_result)
    db.session.add(measurement)
    db.session.commit()
    print(measurement)
    return jsonify(measurement.to_dict())

@app.before_request
def app_context():
    db.create_all()


class Measurement(db.Model):
    __tablename__ = 'measurements'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    lon = db.Column(db.String(120))
    lat = db.Column(db.String(120))
    time_of_measurement = db.Column(db.DateTime, default=datetime.utcnow())
    predicted_result = db.Column(db.String(120))

    def __init__(self, id=None, lon=None, lat=None, time_of_measurement=None, predicted_result=None):
        self.id = id
        self.lon = lon
        self.lat = lat
        if time_of_measurement is not None:
            self.time_of_measurement = time_of_measurement
        else:
            self.time_of_measurement = datetime.utcnow()
        self.predicted_result = predicted_result

    def __repr__(self):
        return f'<Measurement {self.id!r}, {self.lon!r}, {self.lat!r}, {self.time_of_measurement!r}, {self.predicted_result!r}>'

    def to_dict(self):
        return {
            'id': self.id,
            'lon': self.lon,
            'lat': self.lat,
            'time_of_measurement': self.time_of_measurement.isoformat() if self.time_of_measurement else None,
            'predicted_result': self.predicted_result
        }


if __name__ == '__main__':
    app.run(debug=True)
    db.init_app(app)
