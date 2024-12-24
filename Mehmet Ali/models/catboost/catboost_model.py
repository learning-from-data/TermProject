# success rate 98.4


# This uses gpu

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import os

# Load training data
train_feats = np.load('/kaggle/input/lfd-project-dataset/train_feats.npy', allow_pickle=True).item()
train_labels = pd.read_csv('/kaggle/input/lfd-project-dataset/train_labels.csv')
valtest_feats = np.load('/kaggle/input/lfd-project-dataset/valtest_feats.npy', allow_pickle=True).item()

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
labels = train_labels['label'].values

# Initialize CatBoostClassifier
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',
    task_type='GPU',
    random_seed=1
)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)
f1_scores = []

for train_index, val_index in kf.split(train_data):
    X_train, X_val = train_data[train_index], train_data[val_index]
    y_train, y_val = labels[train_index], labels[val_index]

    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)

    val_preds = model.predict(X_val)
    f1 = f1_score(y_val, val_preds, average='macro')
    f1_scores.append(f1)
    print(f"Fold F1 Score: {f1}")

print(f"Average F1 Score: {np.mean(f1_scores)}")

# Fit model on the entire training data
model.fit(train_data, labels, verbose=100)

# Predict on the test set
valtest_preds = model.predict(valtest_data)
valtest_preds = valtest_preds.astype(int)

# Create submission file
# Flatten the predictions if they are multi-dimensional
valtest_preds = valtest_preds.ravel()

submission = pd.DataFrame({
    'ID': np.arange(len(valtest_preds)),
    'Predicted': valtest_preds
})
file_name = "submission_catboost_2.csv"
file_path = os.path.join(os.getcwd(), file_name)


submission.to_csv(file_path, index=False)

print("Submission file created: submission_catboost.csv")
