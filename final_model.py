import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os
import random as r

def load_data(file_path):
    """Load data from a file path, with error handling."""
    try:
        return np.load(file_path, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Load training data
train_feats = load_data('/kaggle/input/lfd-dataset/train_feats.npy')
train_labels = pd.read_csv('/kaggle/input/lfd-dataset/train_labels.csv')
valtest_feats = load_data('/kaggle/input/lfd-dataset/valtest_feats.npy')

# Combine feature vectors into a single feature matrix for training and testing
train_data = np.hstack([
    train_feats['clip_feature'],
    train_feats['dino_feature']
])

valtest_data = np.hstack([
    valtest_feats['clip_feature'],
    valtest_feats['dino_feature']
])

# Labels
y = train_labels['label'].values

# Normalize the features
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
valtest_data = scaler.transform(valtest_data)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=r.seed(1))
f1_scores = []

# SVM model, use sklearn's SVC and it will work with kernel approximation
model = SVC(kernel='rbf', C=500, gamma="scale", random_state=1)

for train_index, val_index in kf.split(train_data):
    X_train, X_val = train_data[train_index], train_data[val_index]
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
model.fit(train_data, y)

# Predict on val/test data
valtest_preds = model.predict(valtest_data)

# Save submission file
file_name = "SVM_submission.csv"
file_path = os.path.join(os.getcwd(), file_name)

submission = pd.DataFrame({
    'ID': np.arange(len(valtest_preds)),
    'Predicted': valtest_preds
})
submission.to_csv(file_path, index=False)

print(f"Submission file created: {file_name}")
