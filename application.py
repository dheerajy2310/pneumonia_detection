import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50 
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from skimage.transform import resize
from flask import Flask, url_for, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import cv2


UPLOAD_FOLDER = 'static/images/'

app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model=load_model('static/models/savedmodel.hdf5')


classes=["Affected with Pneumonia","No existance of Pneumonia"]
def model_predict(img_path,model):
    img=cv2.imread(img_path,0)
    img=img_to_array(img)
    img=resize(img,(150,150))
    img=np.reshape(img,(-1,)+(150, 150, 1))
    return classes[np.argmax(model.predict(img))]

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('home.html')

@app.route('/input',methods=['GET','POST'])
def input():
    return render_template('input.html')

@app.route('/predict',methods=["GET","POST"])
def predict():
        f=request.files['file']
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename))
        f.save(file_path)
        preds=model_predict(file_path,model)
        return preds

if __name__ == '__main__':
    app.run(debug=True)
    