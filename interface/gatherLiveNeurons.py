import requests
import json

def fetch_neuron_data(model_id, layer, index):
    url = 'https://www.neuronpedia.org/api/neuron'
    
    headers = {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'content-type': 'application/json',
        'origin': 'https://www.neuronpedia.org',
        'referer': 'https://www.neuronpedia.org/gemma-2b/6-res-jb/8',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
    }
    
    data = {
        "modelId": model_id,
        "layer": layer,
        "index": index
    }
    
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}, {response.text}"

# Example usage
# result = fetch_neuron_data("gemma-2b", "6-res-jb", "1")
# print(result)

live_neurons = []

for i in range(16384):
    result = fetch_neuron_data("gemma-2b", "6-res-jb", str(i))
    if len(result["activations"]) > 0:
        print(f"{i} has activations")
        live_neurons.append(i)
    else:
        print(f"{i} has no activations")

    with open('live_neurons.json', 'w') as f:
        json.dump(live_neurons, f, indent=4)

