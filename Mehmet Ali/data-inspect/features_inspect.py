import numpy as np

# Load train_feats and valtest_feats with allow_pickle=True
train_feats = np.load('train_feats.npy', allow_pickle=True)
valtest_feats = np.load('valtest_feats.npy', allow_pickle=True)

# Access the stored object
print("Type of train_feats:", type(train_feats))
print("Type of valtest_feats:", type(valtest_feats))

# Check contents if the array contains objects
if isinstance(train_feats, np.ndarray):
    print("Contents of train_feats:", train_feats.item())
if isinstance(valtest_feats, np.ndarray):
    print("Contents of valtest_feats:", valtest_feats.item())
