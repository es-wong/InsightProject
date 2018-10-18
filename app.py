from flask import Flask, render_template, flash, request, redirect, url_for
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from models import imagePathList, fontFamily, featureDimRed, NameVec, model_output, tSNE_vecs, weights




app = Flask(__name__)
#
# @app.route("/")
# def hello():
#     return "Hello World!"
#
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

class ReusableForm(Form):
    fontname = TextField('fontname:', validators=[validators.required()])


@app.route("/process/<fontname>", methods=['GET', 'POST'])
def process(fontname):

    print ("Inside process")

    nameList, scoreList, imgList, inputPath, figPathList  = model_output(fontname, NameVec, featureDimRed,tSNE_vecs,imagePathList, weights)

    imgs = imgList
    names = json.dumps(nameList)
    scores = json.dumps(scoreList)
    input = json.dumps(inputPath)

    print("font name is {}".format(fontname))
    # print(imgs)
    print(figPathList[0])
    return render_template('result.html', fontname=fontname, nameList = nameList,
    names = names, scoreList = scoreList, scores = scores, imgList = imgList, inputPath = inputPath, input = inputPath,
    imgs = imgs, figPathList = figPathList)

@app.route("/", methods=['GET', 'POST'])
def index():
    form = ReusableForm(request.form)
    # model = getmodel()
    # return render_template('index.html',model = model)
    print (form.errors)
    if request.method == 'POST':
        fontname=request.form['fontname']
        print (fontname)

        if form.validate():
            # Save the comment here.
            flash('')

        else:
            flash('All the form fields are required. ')
        print("Ready to redirect")
        return redirect(url_for('process',
                                    fontname=fontname))

    return render_template('index.html', form=form, fullNames = NameVec)
