import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data(file_path):
    """Load data from a file path, with error handling."""
    try:
        return np.load(file_path, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Load training data
train_feats = load_data('train_feats.npy')
train_labels = pd.read_csv('train_labels.csv')
valtest_feats = load_data('valtest_feats.npy')

# Feature keys
feature_keys = ['resnet_feature', 'vit_feature', 'clip_feature', 'dino_feature']

# Labels
y = train_labels['label'].values

# Find the best feature combination
best_f1_score = 0
best_combination = None

for n in range(1, len(feature_keys) + 1):
    for combination in combinations(feature_keys, n):
        # Combine selected features
        train_data = np.hstack([train_feats[key] for key in combination])
        valtest_data = np.hstack([valtest_feats[key] for key in combination])

        # Normalize the features
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        valtest_data = scaler.transform(valtest_data)

        # Dimensionality reduction with PCA
        #pca = PCA(n_components=100)  # Reduce to 100 or fewer components
        #train_data = pca.fit_transform(train_data)
        #valtest_data = pca.transform(valtest_data)

        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(train_data, y, test_size=0.2, random_state=1, stratify=y)

        model = LogisticRegression(max_iter=1000, random_state=1, solver='lbfgs')

        # Train the model
        model.fit(X_train, y_train)

        # Validate the model
        y_pred = model.predict(X_val)
        val_f1_score = f1_score(y_val, y_pred, average='macro')

        print(f"Combination: {combination}, Validation F1 Score: {val_f1_score:.4f}")

        # Update best combination if necessary
        if val_f1_score > best_f1_score:
            best_f1_score = val_f1_score
            best_combination = combination

print(f"Best Combination: {best_combination}, Best Validation F1 Score: {best_f1_score:.4f}")

# Train final model on full training data with the best feature combination
train_data = np.hstack([train_feats[key] for key in best_combination])
valtest_data = np.hstack([valtest_feats[key] for key in best_combination])

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
valtest_data = scaler.transform(valtest_data)



model = LogisticRegression(max_iter=1000, random_state=1, solver='lbfgs')
model.fit(train_data, y)

# Predict on val/test data
valtest_preds = model.predict(valtest_data)

# Save submission file
file_name = "submissionBestCombinationWithoutPCA.csv"
file_path = os.path.join(os.getcwd(), file_name)

submission = pd.DataFrame({
    'ID': np.arange(len(valtest_preds)),
    'Predicted': valtest_preds
})
submission.to_csv(file_path, index=False)

print(f"Submission file created: {file_name}")
