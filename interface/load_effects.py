import torch

# Load all tensors
try:
    top_indices = torch.load("top_is.pt", map_location=torch.device("cpu"))
    top_values = torch.load("top_vs.pt", map_location=torch.device("cpu"))
    reverse_indices = torch.load("reverse_is.pt", map_location=torch.device("cpu"))
    reverse_values = torch.load("reverse_vs.pt", map_location=torch.device("cpu"))
except Exception as e:
    print(f"Error loading PyTorch files: {e}")
    exit(1)

print(f"Loaded tensors with shape: {top_indices.shape}")

while True:
    try:
        user_input = input("Enter a feature number (or 'q' to quit): ")
        if user_input.lower() == "q":
            break

        feature = int(user_input)
        if 0 <= feature < 16364:
            print(f"\nTop indices for feature {feature}:")
            print(top_indices[feature])
            print(f"Top values for feature {feature}:")
            print(top_values[feature])

            print(f"\nReverse indices for feature {feature}:")
            print(reverse_indices[feature])
            print(f"Reverse values for feature {feature}:")
            print(reverse_values[feature])
        else:
            print("Feature out of range. Please enter a number between 0 and 16363.")
    except ValueError:
        print("Invalid input. Please enter a valid integer or 'q' to quit.")
    except KeyboardInterrupt:
        print("\nExiting the program.")
        break

    print("\n" + "-" * 40 + "\n")

print("Program terminated.")
