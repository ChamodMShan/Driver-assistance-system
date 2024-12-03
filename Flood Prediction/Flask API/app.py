from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

app = Flask(__name__)

# Load the model, scaler, and label encoders
model = joblib.load('flood_risk_model_.joblib')
scaler = joblib.load('scaler_.joblib')
label_encoder_district = joblib.load('label_encoder_district_.pkl')
label_encoder_city = joblib.load('label_encoder_city_.pkl')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # POST request
        data = request.get_json()

        if not all(key in data for key in ['Rainfall_(mm)', 'Min_Temp', 'Max_Temp', 'Elevation_(m)']):
            return jsonify({'error': 'Missing required data fields'}), 400

        # Convert input data to a DataFrame
        new_data = pd.DataFrame({
            'Rainfall_(mm)': [data['Rainfall_(mm)']],
            'Min_Temp': [data['Min_Temp']],
            'Max_Temp': [data['Max_Temp']],
            'Elevation_(m)': [data['Elevation_(m)']],
        })

        new_data = new_data.reindex(columns=model.get_booster().feature_names, fill_value=0)

        new_data_scaled = scaler.transform(new_data)

        prediction = model.predict(new_data_scaled)

        prediction = prediction.item()

        if prediction <= 0.33:
            status = 'low'
        elif prediction <= 0.66:
            status = 'medium'
        else:
            status = 'high'

        return jsonify({
            'predicted_flood_risk_probability': prediction,
            'flood_risk_status': status
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
