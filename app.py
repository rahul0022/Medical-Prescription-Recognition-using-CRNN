from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from PIL import Image
from prediction import make_pred
from db_access import get_alternates, get_conditions
import os
UPLOAD_FOLDER = './static/upload'
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
db = SQLAlchemy(app)
medicine_name = ''
alternates = []
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', medicine_name = medicine_name, alternates = alternates)

@app.route('/img_post', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        if 'content' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['content']
        filename = file1.filename
        img = Image.open(file1)
        #print(app.config['UPLOAD_FOLDER'])
        path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename).replace('\\','/')
        img.save(path)
        medicine_name = str(make_pred(path))
        conditions = get_conditions(medicine_name)
        if(len(conditions)==0):
            alternates = ['none']
        else:
            alternates = get_alternates(conditions)

        return render_template('index.html', medicine_name = medicine_name, alternates = alternates)
        
if __name__ == "__main__":
    app.run(debug=True)
