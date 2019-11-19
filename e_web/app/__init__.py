from flask import Flask

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'app/static/images'
app.secret_key = 'vfif'

from app import routes