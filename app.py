from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            int(request.form['worries']),
            int(request.form['activity_fluctuation']),
            int(request.form['social_comparison']),
            int(request.form['concentration_difficulty']),
            int(request.form['easily_distracted']),
            int(request.form['validation_seeking']),
            int(request.form['restlessness_without_sm']),
            int(request.form['sleep_issues']),
            int(request.form['distraction_by_sm']),
            int(request.form['purposeless_use']),
            int(request.form['age']),
            int(request.form['comparison_feelings'])
        ]

        input_data = np.array(data).reshape(1, -1)
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        label = 'Depressed' if prediction == 1 else 'Not Depressed'

        return render_template('index.html', prediction_text=f'Prediction: {label}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)