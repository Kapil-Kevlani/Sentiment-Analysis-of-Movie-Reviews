from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            review = request.form.get('review')
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        if results[0] == 0:
            results = "Negative"
        else:
            results = 'Positive'
        return render_template('home.html', results = results)

if __name__ == '__main__':
    app.run(host = "0.0.0.0", debug = True)

