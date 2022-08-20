import os
from flask import Flask
from flask_uploads import configure_uploads, patch_request_class
from app.forms import photos
from config import Config


app = Flask(__name__)

app.config.from_object(Config)


configure_uploads(app, photos)
# максимальный размер файла, по умолчанию 16MB
patch_request_class(app)

from app import routes, forms


