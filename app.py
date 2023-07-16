import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

# @app.route('/predict_api', methods=['POST'])
# def predict_api():
#     data = request.json['data']
#     print(data)
#     if data['COMMITMENT_FG'] is None:
#         data['COMMITMENT_FG'] = 0
#     elif data['COMMITMENT_FG'] not in [0, 1]:
#         print('Error: COMMITMENT_FG should be either 0 or 1.')
#         return jsonify('Error: COMMITMENT_FG should be either 0 or 1.')
#     print(np.array(list(data.values())).reshape(1, -1))
#     data = np.array(list(data.values())).reshape(1, -1)
#     output = model.predict(data)
#     print(output[0])
#     return jsonify(int(output[0])) 

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        value = list([x for x in request.form.values()])
        if value[5] == 'West Amman':
            value[5] = 0
            value.append(1)
        elif value[5] == 'Irbid' or value[5]=='Jarash' or value[5] == 'Mafraq':
            value[5] = 1
            value.append(0)
        else:
            value[5] = 0
            value.append(0)
        if value[2]=='':
            value[2] = 0
        if value[3]=='':
            value[3] = 0
        print(value)
        data = [ float(x) for x in value]
        final_input = np.array(data).reshape(1, -1)
        print(final_input)
        
        output = model.predict(final_input)[0]
    if output == 1:
        text = 'The customer will leave the company'
    else:
        text = 'The customer will stay'
    
    return render_template('home.html', prediction_text=text, animate=True)
    
    # If the method is GET, render the form
    return render_template('home.html')


if __name__ == "__main__":
    app.run(debug=True)
