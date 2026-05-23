from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Allows website to talk to Flask

# ── Load your saved model and scaler ──
MODEL_PATH  = 'linear_regression_model.pkl'
SCALER_PATH = 'scaler.pkl'

model  = None
scaler = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and scaler loaded successfully!")
else:
    print("⚠️  Model files not found. Download from Colab first.")
    print("    Run in Colab: files.download('linear_regression_model.pkl')")
    print("    Run in Colab: files.download('scaler.pkl')")


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.json

    # Get values from website
    sqft     = float(data['sqft'])
    bedrooms = float(data['bedrooms'])
    fullbath = float(data['fullbath'])
    halfbath = float(data['halfbath'])

    # Prepare input for model
    features = np.array([[sqft, bedrooms, fullbath, halfbath]])
    features_scaled = scaler.transform(features)

    # Predict using YOUR trained model
    price = model.predict(features_scaled)[0]

    return jsonify({
        'price': round(float(price), 2),
        'low':   round(float(price * 0.92), 2),
        'high':  round(float(price * 1.08), 2)
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    print("\n" + "="*40)
    print("  HomeValue AI — Flask Server")
    print("="*40)
    print("  Server: http://127.0.0.1:5000")
    print("  Open index.html with Live Server")
    print("="*40 + "\n")
    app.run(debug=True, port=5000)