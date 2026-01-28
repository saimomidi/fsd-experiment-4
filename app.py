from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load models at startup
print("Loading models...")

with open('iris_logistic_regression.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('target_names.pkl', 'rb') as f:
    target_names = pickle.load(f)

print("Models loaded successfully!")

# Home route
@app.route('/')
def home():
    return render_template(
        'index.html',
        feature_names=feature_names,
        target_names=target_names
    )

# API info route
@app.route('/api/info', methods=['GET'])
def get_info():
    return jsonify({
        'features': feature_names,
        'target_classes': list(target_names),
        'description': 'Iris Flower Classification API'
    })

# Prediction route
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate input
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        features = data.get('features')
        if not features:
            return jsonify({'error': 'Features are required'}), 400

        if len(features) != 4:
            return jsonify({'error': 'Expected 4 features'}), 400

        # Convert to numpy array
        features_array = np.array([features])

        # Make prediction
        prediction = model.predict(features_array)
        prediction_index = int(prediction[0])
        predicted_class = target_names[prediction_index]

        # Get probabilities (if supported)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_array)[0]
            prob_dict = {
                target_names[i]: float(probabilities[i])
                for i in range(len(target_names))
            }
        else:
            prob_dict = None

        # Response
        response = {
            'prediction': predicted_class,
            'prediction_index': prediction_index,
            'input_features': {
                feature_names[i]: features[i]
                for i in range(len(features))
            },
            'probabilities': prob_dict
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
