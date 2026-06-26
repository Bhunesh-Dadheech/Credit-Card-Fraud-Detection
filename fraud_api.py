import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

# Step 1: Load the saved model
model = load_model('fraud_detection_model.h5')

# Step 2: Initialize the Flask app
app = Flask(__name__)

# Step 3: Home route to handle GET requests to the root URL
@app.route('/', methods=['GET'])
def home():
    return "Fraud Detection API is running. Use the /predict endpoint to send POST requests."

# Step 4: Preprocessing function for incoming transaction data
def preprocess_input(data):
    """Preprocess input data to match the model's expected input format."""
    data = np.array(data).reshape(1, -1)  # Reshape data for a single transaction
    return data

# Step 5: Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data
        input_data = request.json['transaction']

        # Preprocess the input data
        preprocessed_data = preprocess_input(input_data)

        # Get the prediction from the model
        prediction = model.predict(preprocessed_data)[0][0]

        # Convert NumPy types to native Python types for JSON serialization
        response = {
            'fraud_probability': float(prediction),  # Convert NumPy float to Python float
            'is_fraud': bool(prediction > 0.5)       # Convert NumPy bool_ to Python bool
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

# Step 6: Run the Flask app locally
if __name__ == '__main__':
    app.run(debug=True)