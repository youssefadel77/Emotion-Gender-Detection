# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 21:18:33 2019

@author: Yahia
"""
from flask import Flask , request , jsonify
from werkzeug import secure_filename
import os
from GenderEmotionClassification import classifayEmotionGender

UPLOAD_FOLDER= ".\images"

app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/all", methods = ['POST'])
def all():
    f = request.files['image']
    filename = secure_filename(f.filename);
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    return classifayEmotionGender(filename)
    

@app.route("/emotion")
def emotion():
    return "emotion calssification"



if __name__ == "__main__":
    app.run(debug=False,host="0.0.0.0",port=5000)