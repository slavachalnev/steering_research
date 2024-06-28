import requests
import json
import atexit
import h5py
import numpy as np

# Load results from new_data.json if it exists
def load_initial_results():
    global results
    try:
        with open('new_data.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []

load_initial_results()  # Call the function to load initial results

def fetch_all_data(start_number):
    global results
    number = start_number

    while True:
        print(number)
        try:
            url = f"https://www.neuronpedia.org/api/feature/gemma-2b/6-res-jb/{number}"
            headers = {
                "accept": "*/*",
                # "accept-language": "en-US,en;q=0.9",
                # "priority": "u=1, i",
                # "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126"',
                # "sec-ch-ua-mobile": "?0",
                # "sec-ch-ua-platform": '"macOS"',
                # "sec-fetch-dest": "empty",
                # "sec-fetch-mode": "cors",
                # "sec-fetch-site": "same-origin",
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an error for bad status codes
            json_data = response.json()
            # results.append({"feature": number, "activations": json_data["activations"], "explanations": json_data["explanations"]})
            results.append({"feature": number, "explanations": list(map(lambda x: x["description"], json_data["explanations"]))})
            
            number += 1  # Move to the next number
        except Exception as e:
            print(e)
            print(results)
            number += 1
            # break  # Stop the loop if there's an error
    return results

# def save_results_on_exit():
#     global results
#     with open('new_data.json', 'w') as f:
#         json.dump(results, f, indent=4)

#     # with h5py.File('data.h5', 'w') as f:
#     #     for i, result in enumerate(results):
#     #         print(list(map(lambda x: x["description"], result["explanations"])))

#     #         grp = f.create_group(f"result_{i}")
#     #         # grp.create_dataset("feature", data=result["index"])
#     #         # grp.create_dataset("activations", data=np.array(result["activations"]))
#     #         grp.create_dataset("explanations", data=np.array(list(map(lambda x: x["description"], result["explanations"])), dtype=h5py.special_dtype(vlen=str)))

# atexit.register(save_results_on_exit)

# fetch_all_data(len(results) + 1)

# import h5py
# import numpy as np

# def load_results(file_path):
#     results = []
#     with h5py.File(file_path, 'r') as f:
#         for key in f.keys():
#             grp = f[key]
#             result = {
#                 "feature": grp["feature"][()],
#                 "activations": grp["activations"][:] if "activations" in grp else None,
#                 "explanations": grp["explanations"][:].astype(str) if "explanations" in grp else None  # Convert back to string
#             }
#             results.append(result)
#     return results

# # Example usage
# results = load_results('data.h5')
# # print(results)




def get_batched_data():
    batch = 0
    global dataset
    dataset = []  # Initialize dataset
    while True:
        print(len(dataset))
        try:
            url = "https://www.neuronpedia.org/api/neurons-offset"
            headers = {
                "accept": "*/*",
                # "accept-language": "en-US,en;q=0.9",
                # "content-type": "application/json",
                # "priority": "u=1, i",
                # "sec-ch-ua": "\"Not/A)Brand\";v=\"8\", \"Chromium\";v=\"126\"",
                # "sec-ch-ua-mobile": "?0",
                # "sec-ch-ua-platform": "\"macOS\"",
                # "sec-fetch-dest": "empty",
                # "sec-fetch-mode": "cors",
                # "sec-fetch-site": "same-origin"
            }
            data = {
                "modelId": "gemma-2b",
                "layer": "6-res-jb",
                "offset": 25 * batch
            }
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # Raise an error for bad status codes
            json = response.json()
            print("new items: ", len(json))
            for feature in json:
                for explanation in feature["explanations"]:
                    dataset.append([explanation["description"], feature["index"]])
            batch += 1
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

def save_dataset_on_exit():
    global dataset
    with open('autointerp.json', 'w') as f:
        json.dump(dataset, f, indent=4)

atexit.register(save_dataset_on_exit)

get_batched_data()