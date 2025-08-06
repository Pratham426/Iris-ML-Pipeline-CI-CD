from flask import Flask, request, jsonify
import joblib
import numpy as np
import json
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load artifacts
MODEL_PATH = "models/iris_model.pkl"
SCALER_PATH = "artifacts/scaler.pkl"
FEATURE_NAMES_PATH = "artifacts/feature_names.json"
CLASS_NAMES_PATH = "artifacts/class_names.json"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURE_NAMES_PATH) as f:
    feature_names = json.load(f)

with open(CLASS_NAMES_PATH) as f:
    class_names = json.load(f)

@app.route('/')
def home():
    """Home endpoint with API instructions"""
    instructions = {
        "API": "Iris Flower Classification API",
        "version": "1.0",
        "endpoints": {
            "/predict": {
                "method": "POST",
                "description": "Predict iris species from measurements",
                "required_fields": feature_names,
                "example_request": {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                }
            }
        },
        "class_names": class_names
    }
    return jsonify(instructions)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    Expected JSON format:
    {
        "sepal_length": float,
        "sepal_width": float,
        "petal_length": float,
        "petal_width": float
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        # Validate input
        missing_fields = [f for f in feature_names if f not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400
        
        # Convert to numpy array
        input_data = np.array([[data[f] for f in feature_names]])
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Format response
        response = {
            "prediction": {
                "class_id": int(prediction[0]),
                "class_name": class_names[int(prediction[0])]
            },
            "probabilities": dict(zip(class_names, [float(p) for p in probabilities])),
            "input_features": dict(zip(feature_names, [float(x) for x in input_data[0]]))
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
