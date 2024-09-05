from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('crop_recommend.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(data)
    return render_template('index.html', prediction_text=f'Recommended Crop: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)

