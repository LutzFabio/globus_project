from flask import Flask
#import tensorflow as tf
#from tensorflow_core.keras import models

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'app/static/images'
app.secret_key = 'vfif'

#best_model = models.load_model('app/static/models/model_own.h5')

from app import routes