from app import app
from flask import render_template
from app.forms import UploadForm, photos
from app.classification import get_prediction
import cv2, os

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    name = None
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        # file_url = photos.url(filename)
        
        name = get_prediction()
        # filename1 = photos.save(img)
        # file_url = photos.url(filename1)
    # else:
    #     file_url = None
    return  render_template('index.html', form=form, file_url=name)