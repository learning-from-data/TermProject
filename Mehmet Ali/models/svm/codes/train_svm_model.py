import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

# Load training data
train_feats = np.load("train_feats.npy", allow_pickle=True).item()  # Assuming this is a dictionary
train_labels = pd.read_csv("train_labels.csv")
train_labels = train_labels['label']  # Assuming 'label' column has the target values

# Concatenate all feature types for training
train_data = np.hstack([
    train_feats['resnet_feature'],
    train_feats['vit_feature'],
    train_feats['clip_feature'],
    train_feats['dino_feature']
])

# Load validation/test data
valtest_feats = np.load("valtest_feats.npy", allow_pickle=True).item()
valtest_data = np.hstack([
    valtest_feats['resnet_feature'],
    valtest_feats['vit_feature'],
    valtest_feats['clip_feature'],
    valtest_feats['dino_feature']
])

# Define SVM model pipeline with preprocessing
svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=False, random_state=42))

# Train the SVM
svm_pipeline.fit(train_data, train_labels)

# Predict on validation/test data
predictions = svm_pipeline.predict(valtest_data)

# Prepare submission file
submission = pd.DataFrame({
    "ID": np.arange(len(predictions)),
    "Predicted": predictions
})
submission.to_csv("submission_svm.csv", index=False)
print("Submission file saved as submission.csv.")
