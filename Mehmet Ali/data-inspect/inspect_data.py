import numpy as np
import pandas as pd

# File paths
train_feats_path = 'train_feats.npy'
train_labels_path = 'train_labels.csv'
valtest_feats_path = 'valtest_feats.npy'
samplesubmission_path = 'samplesubmission.csv'

# Load data (set allow_pickle=True for .npy files)
train_feats = np.load(train_feats_path, allow_pickle=True)
train_labels = pd.read_csv(train_labels_path)
valtest_feats = np.load(valtest_feats_path, allow_pickle=True)
samplesubmission = pd.read_csv(samplesubmission_path)

# Inspect dimensions and basic properties
data_info = {
    "train_feats_shape": train_feats.shape,
    "train_labels_shape": train_labels.shape,
    "valtest_feats_shape": valtest_feats.shape,
    "sample_submission_shape": samplesubmission.shape,
    "train_labels_head": train_labels.head(),
    "train_labels_distribution": train_labels.iloc[:, 1].value_counts().to_dict()  # Assuming labels are in the second column
}

print(data_info)
