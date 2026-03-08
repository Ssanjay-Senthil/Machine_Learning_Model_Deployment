from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load ML model and scaler
model = joblib.load("gradient_boosting_model.joblib")
scaler = joblib.load("minmax_scaler.joblib")

# Crop label mapping
target_names = {
    0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut',
    5: 'coffee', 6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans',
    10: 'lentil', 11: 'maize', 12: 'mango', 13: 'mothbeans', 14: 'mungbean',
    15: 'muskmelon', 16: 'orange', 17: 'papaya', 18: 'pigeonpeas',
    19: 'pomegranate', 20: 'rice', 21: 'watermelon'
}

# Home route
@app.route("/")
def home():
    return jsonify({
        "message": "Crop Recommendation API is running",
        "endpoint": "/predict"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:

        data = request.get_json()

        required_features = [
            "N", "P", "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall"
        ]

        # Check missing values
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

        # Convert input to dataframe
        input_data = pd.DataFrame([data])[required_features]

        # Scale input
        scaled_data = scaler.transform(input_data)

        # Predict crop
        prediction = model.predict(scaled_data)

        crop_id = int(prediction[0])
        crop_name = target_names.get(crop_id, "Unknown")

        return jsonify({
            "predicted_crop": crop_name,
            "input_data": data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
