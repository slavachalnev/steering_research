import os
import time
import threading
import requests
from flask import Flask, jsonify, request
from dotenv import load_dotenv

# from flask_cors import CORS

app = Flask(__name__)
# CORS(app)
# CORS(app, resources={r"/predict": {"origins": "*", "allow_headers": ["Content-Type"]}})

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API token from the environment variables
api_token = os.getenv("REPLICATE_API_TOKEN")

app = Flask(__name__)

url = "https://api.replicate.com/v1/predictions"
headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
data = {
    "version": "806d4b25f02fbffee8076a34423ecdf8e261774c75adde941e17ed3a49457712",
    "input": {
        "prompt": "The best city",
        "steering": "[[10138, 40]]",
        "n_samples": 1,
        "batch_size": 1,
        "max_new_tokens": 1,
    },
}


def send_request():
    response = requests.post(url, headers=headers, json=data)
    response_data = response.json()

    # Extract the prediction ID for polling
    prediction_id = response_data["id"]
    status = response_data["status"]

    # Polling the API until the prediction is complete
    while status not in ["succeeded", "failed"]:
        time.sleep(5)  # Wait for 5 seconds before polling again
        poll_url = f"{url}/{prediction_id}"
        poll_response = requests.get(poll_url, headers=headers)
        poll_response_data = poll_response.json()
        status = poll_response_data["status"]
        print(f"Prediction status: {status}")

    # Print the final response
    if status == "succeeded":
        print("Prediction succeeded:", poll_response_data["output"])
    else:
        print("Prediction failed:", poll_response_data["error"])


def keep_alive():
    while True:
        send_request()
        time.sleep(60)  # Wait for 1 minute before sending the next request


# Start a background thread to keep the API alive
threading.Thread(target=keep_alive, daemon=True).start()


@app.route("/predict", methods=["OPTIONS"])
def handle_options():
    response = app.make_default_options_response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/predict", methods=["POST"])
def predict():
    custom_data = request.json
    response = requests.post(url, headers=headers, json=custom_data)
    response_data = response.json()

    # Extract the prediction ID for polling
    prediction_id = response_data["id"]
    status = response_data["status"]

    # Polling the API until the prediction is complete
    while status not in ["succeeded", "failed"]:
        time.sleep(5)  # Wait for 5 seconds before polling again
        poll_url = f"{url}/{prediction_id}"
        poll_response = requests.get(poll_url, headers=headers)
        poll_response_data = poll_response.json()
        status = poll_response_data["status"]
        print(f"Prediction status: {status}")

    # Return the final response
    if status == "succeeded":
        response = jsonify(
            {"status": "succeeded", "output": poll_response_data["output"]}
        )
    else:
        response = (
            jsonify({"status": "failed", "error": poll_response_data["error"]}),
            400,
        )

    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"

    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
