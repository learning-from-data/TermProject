# This uses gpu

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import os

# Load training data
train_feats = np.load('train_feats.npy', allow_pickle=True).item()
train_labels = pd.read_csv('train_labels.csv')

# Combine all feature types into a single feature matrix
features = np.hstack([
    train_feats['resnet_feature'],
    train_feats['vit_feature'],
    train_feats['clip_feature'],
    train_feats['dino_feature']
])

labels = train_labels['Label'].values

# Initialize LightGBM dataset
kf = KFold(n_splits=5, shuffle=True, random_state=1)

def train_and_evaluate(X, y, params, n_folds=5):
    f1_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            verbose_eval=100
        )

        y_pred = np.argmax(model.predict(X_val, num_iteration=model.best_iteration), axis=1)
        f1 = f1_score(y_val, y_pred, average='macro')
        f1_scores.append(f1)

    return model, np.mean(f1_scores)

# Define LightGBM parameters
params = {
    'objective': 'multiclass',
    'num_class': 10,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'device': 'gpu',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

# Train model using 5-fold cross-validation
final_model, mean_f1 = train_and_evaluate(features, labels, params)
print(f'Mean F1 Score: {mean_f1}')

# Load validation/test features
valtest_feats = np.load('valtest_feats.npy', allow_pickle=True).item()
valtest_features = np.hstack([
    valtest_feats['resnet_feature'],
    valtest_feats['vit_feature'],
    valtest_feats['clip_feature'],
    valtest_feats['dino_feature']
])

# Predict on valtest data
predictions = np.argmax(final_model.predict(valtest_features, num_iteration=final_model.best_iteration), axis=1)

# Create submission file
submission = pd.DataFrame({'ID': np.arange(len(predictions)), 'Predicted': predictions})
submission.to_csv('predictions.csv', index=False)
print("Submission file 'predictions.csv' created successfully.")
