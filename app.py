from flask import Flask, render_template, request
import pandas as pd
import joblib
import json

app = Flask(__name__)

# Load model + scaler
model = joblib.load('model_terbaik.joblib')
scaler = joblib.load('scaler.joblib')

# Load metadata
with open("model_metadata.json", "r") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {}

        for col in FEATURES:
            val = request.form.get(col, "")
            if val.strip() == "":
                return f"Kolom '{col}' harus diisi!"
            data[col] = val

        # Encoding kategori
        data['gender'] = 1 if data['gender'].lower() == 'male' else 0
        data['smoker'] = 1 if data['smoker'].lower() == 'ya' else 0
        data['alcohol'] = 1 if data['alcohol'].lower() == 'ya' else 0
        data['family_history'] = 1 if data['family_history'].lower() == 'ya' else 0

        numeric_cols = [
            'age','bmi','daily_steps','sleep_hours','water_intake_l',
            'calories_consumed','resting_hr','systolic_bp','diastolic_bp','cholesterol'
        ]

        for col in numeric_cols:
            data[col] = float(data[col])

        X_new = pd.DataFrame([data], columns=FEATURES)
        X_scaled = scaler.transform(X_new)

        # Probability
        proba = model.predict_proba(X_scaled)[0][1]

        # Threshold 0.5
        result = "Risiko Tinggi" if proba >= 0.5 else "Risiko Rendah"

        return render_template(
            'index.html', 
            prediction=result, 
            prob=proba,
            old=data
        )

    except Exception as e:
        return f"Terjadi error: {e}"

if __name__ == "__main__":
    app.run(debug=True)