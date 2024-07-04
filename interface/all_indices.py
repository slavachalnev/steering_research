import torch
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load all tensors
try:
    tensor_data_indices = torch.load('all_indices.pt', map_location=torch.device('cpu'))
    tensor_data_values = torch.load('all_values.pt', map_location=torch.device('cpu'))
    cos_sim_indices = torch.load('cosine_sim_indices.pt', map_location=torch.device('cpu'))
    cos_sim_values = torch.load('cosine_sim_values.pt', map_location=torch.device('cpu'))
except Exception as e:
    print(f"Error loading PyTorch files: {e}")
    exit(1)

def normalize_values(values):
    # Ensure all values are non-negative
    min_val = min(values)
    shifted_values = [v - min_val for v in values]
    
    # Calculate the sum of all shifted values
    total = sum(shifted_values)
    
    # If all values are the same (total is 0), return equal proportions
    if total == 0:
        return [1.0 / len(values)] * len(values)
    
    # Normalize values so they sum to 1 while maintaining relative scales
    return [v / total for v in shifted_values]

@app.route('/get_data', methods=['OPTIONS'])
def handle_get_data_options():
    response = app.make_default_options_response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route('/get_data', methods=['GET'])
def get_data():
    index = request.args.get('index', type=int)
    data_type = request.args.get('type', type=str)

    if index is None or data_type is None:
        return jsonify({"error": "Missing parameters"}), 400

    if data_type == "tensor":
        indices = tensor_data_indices[index].tolist()
        values = tensor_data_values[index].tolist()
    elif data_type == "cosine":
        indices = cos_sim_indices[index].tolist()
        values = cos_sim_values[index].tolist()
    else:
        return jsonify({"error": "Invalid data type"}), 400

    normalized_values = normalize_values(values)

    response = jsonify({
        "indices": indices,
        "values": normalized_values
    })

    # Add CORS headers to the response
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"

    return response

if __name__ == '__main__':
    app.run(debug=True)