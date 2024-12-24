# This uses gpu

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
import xgboost as xgb

# Load data
train_feats = np.load('train_feats.npy', allow_pickle=True).item()
train_labels = pd.read_csv('train_labels.csv')
valtest_feats = np.load('valtest_feats.npy', allow_pickle=True).item()

# Concatenate all feature types for training and validation/test sets
X_train = np.hstack([
    train_feats['resnet_feature'], 
    train_feats['vit_feature'], 
    train_feats['clip_feature'], 
    train_feats['dino_feature']
])

X_valtest = np.hstack([
    valtest_feats['resnet_feature'], 
    valtest_feats['vit_feature'], 
    valtest_feats['clip_feature'], 
    valtest_feats['dino_feature']
])

# Extract labels
y_train = train_labels['label'].values

# Prepare cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)
f1_scores = []

# Cross-validation loop
for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, tree_method='gpu_hist', predictor='gpu_predictor')
    model.fit(X_train_fold, y_train_fold)

    y_val_pred = model.predict(X_val_fold)
    f1 = f1_score(y_val_fold, y_val_pred, average='macro')
    f1_scores.append(f1)

# Print average F1 score
print(f"Average F1 Score from 5-fold CV: {np.mean(f1_scores):.4f}")

# Train final model on the entire dataset
final_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, tree_method='gpu_hist', predictor='gpu_predictor')
final_model.fit(X_train, y_train)

# Predict on validation/test set
valtest_preds = final_model.predict(X_valtest)

# Create submission file
submission = pd.DataFrame({
    'ID': np.arange(len(valtest_preds)),
    'Predicted': valtest_preds
})
submission.to_csv('submission_XGBoost.csv', index=False)

print("Submission file saved as submission.csv")
