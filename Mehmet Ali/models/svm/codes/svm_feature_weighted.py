import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import os

# Load training data
train_feats = np.load('train_feats.npy', allow_pickle=True).item()
train_labels = pd.read_csv('train_labels.csv')
valtest_feats = np.load('valtest_feats.npy', allow_pickle=True).item()

# Combine feature vectors into a single feature matrix for training and testing
def safe_hstack(features, keys):
    valid_features = [features[key] for key in keys if key in features and features[key] is not None]
    return np.hstack(valid_features) if valid_features else None

feature_keys = ['resnet_feature', 'vit_feature', 'clip_feature', 'dino_feature']

# Safely stack features
train_data = safe_hstack(train_feats, feature_keys)
valtest_data = safe_hstack(valtest_feats, feature_keys)

if train_data is None or valtest_data is None:
    raise ValueError("One or more feature arrays are missing or invalid.")

# Labels
y = train_labels['label'].values

# Feature indexing for important features (manual ranges for positional indexing)
feature_ranges = {
    'clip_feature': range(0, train_feats['clip_feature'].shape[1]),
    'dino_feature': range(
        train_feats['clip_feature'].shape[1], 
        train_feats['clip_feature'].shape[1] + train_feats['dino_feature'].shape[1]
    ),
    # Add similar ranges for other feature types
}

# Manually map important features to indices
important_feature_indices = [
    feature_ranges['clip_feature'][471],
    feature_ranges['clip_feature'][115],
    feature_ranges['clip_feature'][102],
    feature_ranges['clip_feature'][465],
    feature_ranges['dino_feature'][447],
    # Continue adding other indices if needed
]

# Select these features
train_data_selected = train_data[:, important_feature_indices]
valtest_data_selected = valtest_data[:, important_feature_indices]

# Feature scaling
scaler = StandardScaler()
train_data_selected = scaler.fit_transform(train_data_selected)
valtest_data_selected = scaler.transform(valtest_data_selected)

# Compute class weights for imbalanced data
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}

# SVC with class weights
model = SVC(
    class_weight=class_weights_dict,
    random_state=42
)

# 5-fold cross-validation with the model
kf = KFold(n_splits=5, shuffle=True, random_state=1)
f1_scores = []

for train_index, val_index in kf.split(train_data_selected):
    X_train, X_val = train_data_selected[train_index], train_data_selected[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Train the model
    model.fit(X_train, y_train)

    # Validate the model
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')
    f1_scores.append(f1)

    print(f"Fold F1 Score: {f1:.4f}")

print(f"Mean F1 Score: {np.mean(f1_scores):.4f}")

# Train final model on full training data
model.fit(train_data_selected, y)

# Predict on val/test data
valtest_preds = model.predict(valtest_data_selected)

# Save submission file
file_name = "submission_optimized_svm.csv"
file_path = os.path.join(os.getcwd(), file_name)

submission = pd.DataFrame({
    'ID': np.arange(len(valtest_preds)),
    'Predicted': valtest_preds
})
submission.to_csv(file_path, index=False)

print(f"Submission file created: {file_name}")
