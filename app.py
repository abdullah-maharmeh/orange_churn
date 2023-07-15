import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model = pickle.load(open('churnmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    if data['COMMITMENT_FG'] is None:
        data['COMMITMENT_FG'] = 0
    elif data['COMMITMENT_FG'] not in [0, 1]:
        print('Error: COMMITMENT_FG should be either 0 or 1.')
        return jsonify('Error: COMMITMENT_FG should be either 0 or 1.')
    print(np.array(list(data.values())).reshape(1, -1))
    data = np.array(list(data.values())).reshape(1, -1)
    output = model.predict(data)
    print(output[0])
    return jsonify(int(output[0]))  # Convert the output to an integer

if __name__ == "__main__":
    app.run(debug=True)
