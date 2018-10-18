from flask import Flask, render_template, flash, request, redirect, url_for
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from models import imagePathArray, fontFamily, featureDimRed, fullNames
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json



app = Flask(__name__)

app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

class ReusableForm(Form):
    fontname = TextField('fontname:', validators=[validators.required()])


def model_output(fontname, featureDimRed, fontFamily):
    '''
    This function outputs the names of similar fonts with their similariy values
    '''
    idxArray = []
    idxArray = [idx for idx,label in enumerate(fontFamily) if fontname in label]

    modelOutput = []
    for idx in idxArray:
        modelOutput.append(cosine_similarity([featureDimRed[idx]],featureDimRed))

    modelOutput = np.asarray(modelOutput)

    predictions = []
    for index,output in enumerate(modelOutput):
        predictions.append((-modelOutput[index]).argsort()[:5][0][1:5])

    predictions = np.asarray(predictions)

    outDict1 = dict(zip(idxArray,predictions))

    labelMatch = []
    labelPredict = []

    for key, value in outDict1.items():
        labelMatch.append(fontFamily[key])
        labelPredict.append([fontFamily[value[:]]])

    outDict2 = dict(zip(labelMatch,labelPredict))
    outDict3 = dict(zip(labelMatch,modelOutput[0][0][predictions[:]]))

    outFin1 = []
    outFin2 = []
    for key in outDict2.keys():
        outFin1.append(outDict2[key][:])
        outFin2.append(outDict3[key][:])

    return zip([thing for thing in outFin1[0]],
                list(outFin2))


@app.route("/process/<fontname>", methods=['GET', 'POST'])
def process(fontname):

    print ("Inside process")
    result = model_output(fontname,featureDimRed,fontFamily)

    print("font name is {}".format(fontname))
    return render_template('result.html', fontname=fontname, outFin1=result)

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

    return render_template('index.html', form=form, fullNames = fullNames)
