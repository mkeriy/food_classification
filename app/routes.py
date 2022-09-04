from app import app
from flask import render_template, make_response
from app.forms import UploadForm, photos
from app.classification import get_prediction
import cv2, os

@app.route('/', methods=['POST'])
def upload_file():
    name = None
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        name = get_prediction()
        return name
    else:
        res = make_response('<h1> Bad Request </h1>', 400)
        return res
    

@app.route('/', methods=['GET'])
def send_prediction():
    form = UploadForm()
    return  render_template('index.html', form=form)