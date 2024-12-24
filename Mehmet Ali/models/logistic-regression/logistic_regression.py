import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import os

# Load training data
train_feats = np.load('train_feats.npy', allow_pickle=True).item()
train_labels = pd.read_csv('train_labels.csv')
valtest_feats = np.load('valtest_feats.npy', allow_pickle=True).item()

# Combine feature vectors into a single feature matrix for training and testing
train_data = np.hstack([
    train_feats['resnet_feature'],
    train_feats['vit_feature'],
    train_feats['clip_feature'],
    train_feats['dino_feature']
])

valtest_data = np.hstack([
    valtest_feats['resnet_feature'],
    valtest_feats['vit_feature'],
    valtest_feats['clip_feature'],
    valtest_feats['dino_feature']
])

# Extract labels
train_labels = train_labels['label'].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)
f1_scores = []
valtest_preds = np.zeros(valtest_data.shape[0])

# Logistic Regression model
for train_index, val_index in kf.split(train_data):
    X_train, X_val = train_data[train_index], train_data[val_index]
    y_train, y_val = train_labels[train_index], train_labels[val_index]

    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)

    # Validation predictions
    val_preds = model.predict(X_val)
    f1 = f1_score(y_val, val_preds, average='macro')
    f1_scores.append(f1)

    print(f"Fold F1 Score: {f1}")

# Average F1 score
print(f"Average F1 Score: {np.mean(f1_scores)}")

# Train final model on entire training data
final_model = LogisticRegression(max_iter=1000, solver='liblinear')
final_model.fit(train_data, train_labels)

# Predict on valtest data
valtest_preds = final_model.predict(valtest_data)

# Save predictions to CSV
file_name = "submission_logistic_regression.csv"
file_path = os.path.join(os.getcwd(), file_name)

submission = pd.DataFrame({
    'ID': np.arange(len(valtest_preds)),
    'Predicted': valtest_preds
})
submission.to_csv(file_path, index=False)

print(f"Submission file created: {file_name}")
