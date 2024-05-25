from datetime import datetime
from database import db

class Measurement(db.Model):
    __tablename__ = 'measurements'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    lon = db.Column(db.String(120))
    lat = db.Column(db.String(120))
    time_of_measurement = db.Column(db.DateTime, default=datetime.utcnow)
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
