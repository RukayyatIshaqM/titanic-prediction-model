import numpy as np
from flask import Flask, render_template
import pickle
from pathlib import Path

app = Flask(__name__)

# Load the trained model
model_path = Path("model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/')
def home():
    return render_template('index.html')

app.route('/predict, methods=['POST'])
def predict():
    # Get input data from the form
    pclass = int (request.form['pclass'])
    sex = int(request.form['sex'])
    age = float(request.form['age'])
    sibsp = int(request.form['sibsp'])
    parch = int(request.form['parch'])
    fare = float(request.form['fare'])

    final_fearture = [pclass, sex, age,sibsp, parch, fare]

    # Make prediction using the loaded model
    prediction = model.predict(np.array(final_feature).reshape(1, -1))
    return render_template('index.html, result = survived)

    return render_template('results.html', results='survived',if prediction=prediction[0]==1 else 'not survived')


if __name__ == '__main__':
    app.run(debug=True)
