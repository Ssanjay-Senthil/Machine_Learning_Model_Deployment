from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("gradient_boosting_model.joblib")
scaler = joblib.load("minmax_scaler.joblib")

# Required features (same order as training)
FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

# -----------------------------
# Home / Health check endpoint
# -----------------------------
@app.route("/")
def home():
    return jsonify({
        "message": "Crop Recommendation API is running",
        "endpoint": "/predict",
        "required_features": FEATURES
    })


# -----------------------------
# Crop prediction endpoint
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:

        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON input provided"}), 400

        # Check missing features
        for feature in FEATURES:
            if feature not in data:
                return jsonify({
                    "error": f"Missing feature: {feature}"
                }), 400

        # Convert input to dataframe
        input_data = pd.DataFrame([data])[FEATURES]

        # Scale input
        scaled_data = scaler.transform(input_data)

        # Predict crop
        prediction = model.predict(scaled_data)
        predicted_crop = prediction[0]

        # Optional improvement → prediction probability
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(scaled_data)
            confidence = float(np.max(probs))
        else:
            confidence = None

        return jsonify({
            "predicted_crop": predicted_crop,
            "confidence": confidence,
            "input_data": data
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
