import json

# Read the existing JSON file
with open("./autointerp.json", "r") as file:
    data = json.load(file)

# Convert the list of lists to a dictionary
new_data = {item[1]: item[0] for item in data}

# Write the new dictionary to the JSON file
with open("./new_autointerp.json", "w") as file:
    json.dump(new_data, file, indent=4)

print("Conversion complete. The file has been updated.")
