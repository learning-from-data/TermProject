import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load train_feats and valtest_feats
train_feats = np.load('train_feats.npy', allow_pickle=True).item()
valtest_feats = np.load('valtest_feats.npy', allow_pickle=True).item()

# Extract individual features and concatenate them
X_train_resnet = train_feats['resnet_feature']
X_train_vit = train_feats['vit_feature']
X_train_clip = train_feats['clip_feature']
X_train_dino = train_feats['dino_feature']

# Concatenate all features into a unified representation
X_train_combined = np.concatenate(
    [X_train_resnet, X_train_vit, X_train_clip, X_train_dino], axis=1
)

# Load labels
train_labels = pd.read_csv('train_labels.csv')
y_train = train_labels['label'].values

# Train-validation split
X_train, X_val, y_train_split, y_val_split = train_test_split(
    X_train_combined, y_train, test_size=0.2, random_state=42
)

# Validation/Test features
X_test_resnet = valtest_feats['resnet_feature']
X_test_vit = valtest_feats['vit_feature']
X_test_clip = valtest_feats['clip_feature']
X_test_dino = valtest_feats['dino_feature']

# Concatenate all test features
X_test_combined = np.concatenate(
    [X_test_resnet, X_test_vit, X_test_clip, X_test_dino], axis=1
)

# Outputs
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test_combined.shape)
print("y_train_split distribution:", np.unique(y_train_split, return_counts=True))
print("y_val_split distribution:", np.unique(y_val_split, return_counts=True))
