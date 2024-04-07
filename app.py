from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

model = joblib.load('random_forest_regressor.pkl')
scaler = joblib.load('standard_scaler.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    print(data)
    if data['day_or_night'] == 'Day':
        data['day_or_night'] = 1
    else:
        data['day_or_night'] = 0

    new_data = [[float(data[key]) for key in data]]

    new_data_scaled = scaler.transform(new_data)

    prediction = model.predict(new_data_scaled)

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
