import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os

# Load training data
train_feats = np.load('../train-files/train_feats.npy', allow_pickle=True).item()
train_labels = pd.read_csv('../train-files/train_labels.csv')
valtest_feats = np.load('../train-files/valtest_feats.npy', allow_pickle=True).item()

# Combine feature vectors into a single feature matrix for training and testing
train_data = np.hstack([
    train_feats['resnet_feature']
])

valtest_data = np.hstack([
    valtest_feats['resnet_feature']
])

# Labels
y = train_labels['label'].values

# Normalize the features
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
valtest_data = scaler.transform(valtest_data)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)
f1_scores = []

# SVM model
model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42, class_weight='balanced')

# File to save F1 scores
f1_file_name = "f1_scores_svm_normalized_rbf.txt"
f1_file_path = os.path.join(os.getcwd(), '..', 'f1-scores', f1_file_name)

with open(f1_file_path, 'w') as f:
    for train_index, val_index in kf.split(train_data):
        X_train, X_val = train_data[train_index], train_data[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Train the model
        model.fit(X_train, y_train)

        # Validate the model
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='macro')
        f1_scores.append(f1)

        # Write F1 score to file
        f.write(f"Fold F1 Score: {f1:.4f}\n")
        print(f"Fold F1 Score: {f1:.4f}")

    # Write mean F1 score to file
    mean_f1 = np.mean(f1_scores)
    f.write(f"Mean F1 Score: {mean_f1:.4f}\n")
    print(f"Mean F1 Score: {mean_f1:.4f}")

# Train final model on full training data
model.fit(train_data, y)

# Predict on val/test data
valtest_preds = model.predict(valtest_data)

# Save submission file
file_name = "submission_svm_normalized_rbf_class.csv"
file_path = os.path.join(os.getcwd(), file_name)

submission = pd.DataFrame({
    'ID': np.arange(len(valtest_preds)),
    'Predicted': valtest_preds
})
submission.to_csv(file_path, index=False)

print(f"Submission file created: {file_name}")
print(f"F1 scores file created: {f1_file_name}")
