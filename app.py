from flask import Flask,render_template,url_for,request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

app=Flask(__name__)
Swagger(app)

mnb = pickle.load(open('/Users/gauravlohkna/Downloads/Part-3/Model_sentiment_mnb_new.pkl','rb'))
countVector = pickle.load(open('/Users/gauravlohkna/Downloads/Part-3/Model_sentiment_countVect_new.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        Reviews = request.form['Reviews']
        data = [Reviews]
        Vector = countVector.transform(data)
        predicted_value = mnb.predict(Vector)
    return render_template('result.html',prediction = predicted_value)



if __name__ == '__main__':
  app.run(host='0.0.0.0', port=80)