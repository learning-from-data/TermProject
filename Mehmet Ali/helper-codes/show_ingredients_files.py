import numpy as np

# Replace 'path_to_file.npy' with the actual path to your .npy file
npy_file_path = 'train_feats.npy'
output_txt_path = 'ingredients.txt'  # The file where we'll save the information

# Load the .npy file
data = np.load(npy_file_path, allow_pickle=True)


# Open the output file for writing
with open(output_txt_path, 'w') as f:
    # Write basic information about the loaded data
    f.write(f"Data Shape: {data.shape}\n")
    f.write(f"Data Type: {data.dtype}\n\n")

    # Write the entire data contents
    f.write("Data Contents:\n")
    f.write(str(data))
