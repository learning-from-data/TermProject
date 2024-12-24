import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump

# Load train_feats and valtest_feats
train_feats = np.load('train_feats.npy', allow_pickle=True).item()
valtest_feats = np.load('valtest_feats.npy', allow_pickle=True).item()

# Extract and concatenate features
X_train_resnet = train_feats['resnet_feature']
X_train_vit = train_feats['vit_feature']
X_train_clip = train_feats['clip_feature']
X_train_dino = train_feats['dino_feature']

X_train_combined = np.concatenate(
    [X_train_resnet, X_train_vit, X_train_clip, X_train_dino], axis=1
)

# Validation/Test features
X_test_resnet = valtest_feats['resnet_feature']
X_test_vit = valtest_feats['vit_feature']
X_test_clip = valtest_feats['clip_feature']
X_test_dino = valtest_feats['dino_feature']

X_test_combined = np.concatenate(
    [X_test_resnet, X_test_vit, X_test_clip, X_test_dino], axis=1
)

# Load labels
train_labels = pd.read_csv('train_labels.csv')
y_train = train_labels['label'].values

# Split data for validation
X_train, X_val, y_train_split, y_val_split = train_test_split(
    X_train_combined, y_train, test_size=0.2, random_state=42
)

# Train a baseline Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train_split)

# Validate the model
y_val_pred = rf_model.predict(X_val)
print("Validation Results:\n", classification_report(y_val_split, y_val_pred))

# Predict on test data
y_test_pred = rf_model.predict(X_test_combined)

# Generate submission file
submission = pd.DataFrame({
    'ID': valtest_feats['idx'],
    'label': y_test_pred
})
submission.to_csv('submission.csv', index=False)

# Save the model
dump(rf_model, 'random_forest_model.joblib')

print("Submission file 'submission.csv' created!")
