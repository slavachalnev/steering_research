import json

# # Load the JSON data from the file
# with open('./new_data.json', 'r') as file:
#     data = json.load(file)

# items = list(map(lambda x: "".join(x["explanations"]), data))

# print(len(data))

# def search_features(data, search_term):
#     results = []
#     for i, item in enumerate(items):
#         if search_term.lower() in item.lower():
#             results.append([item, i])
#     return results

# if __name__ == "__main__":
#     while True:
#         search_term = input("Enter search term: ")
#         results = search_features(data, search_term)
#         print("Search results:", results)


# Load the JSON data from the file
with open('./autointerp.json', 'r') as file:
    data = json.load(file)
# items = list(map(lambda x: x[0], data))  # Extract descriptions

print(len(data))

def search_features(data, search_term):
    results = []
    for i, item in enumerate(data):
        if search_term.lower() in item[0].lower():
            results.append(item)
    return results

if __name__ == "__main__":
    while True:
        search_term = input("Enter search term: ")
        results = search_features(data, search_term)
        print("Search results:")
        for result in results:
            print(f"{result[1]} - {result[0]}")

