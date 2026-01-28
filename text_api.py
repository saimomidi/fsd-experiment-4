import requests
import json

# Base URL of the running Flask API
BASE_URL = "http://localhost:5000"

# Sample input features for Iris dataset
# Order should match: [sepal_length, sepal_width, petal_length, petal_width]
payload = {
    "features": [5.1, 3.5, 1.4, 0.2]
}

try:
    # Send POST request to prediction endpoint
    response = requests.post(
        f"{BASE_URL}/api/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    # Print status and response
    print("Status Code:", response.status_code)
    print("Response:")
    print(json.dumps(response.json(), indent=2))

except Exception as e:
    print("Error while testing API:", e)
