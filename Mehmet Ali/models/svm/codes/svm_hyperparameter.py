import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
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

# Labels
y = train_labels['label'].values

# Feature selection: Remove low-variance features
selector = VarianceThreshold(threshold=0.01)
train_data = selector.fit_transform(train_data)
valtest_data = selector.transform(valtest_data)

# Dimensionality reduction with PCA
pca = PCA(n_components=0.95)
train_data = pca.fit_transform(train_data)
valtest_data = pca.transform(valtest_data)

# SMOTE for data augmentation
smote = SMOTE(random_state=1)
train_data, y = smote.fit_resample(train_data, y)

# Define a parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['linear', 'poly', 'rbf']
}

# Perform hyperparameter tuning for each SVM
svm_base = SVC(class_weight='balanced', random_state=1, probability=True)
grid_search = GridSearchCV(svm_base, param_grid, scoring='f1_macro', cv=3, n_jobs=-1)
grid_search.fit(train_data, y)

# Best parameters from grid search
print(f"Best Parameters: {grid_search.best_params_}")

# Use the best parameters to create SVM models
best_params = grid_search.best_params_
svm_linear = SVC(**{**best_params, 'kernel': 'linear'}, probability=True, class_weight='balanced')
svm_poly = SVC(**{**best_params, 'kernel': 'poly'}, probability=True, class_weight='balanced')
svm_rbf = SVC(**{**best_params, 'kernel': 'rbf'}, probability=True, class_weight='balanced')

# Create a voting classifier
voting_model = VotingClassifier(estimators=[
    ('svm_linear', svm_linear),
    ('svm_poly', svm_poly),
    ('svm_rbf', svm_rbf)
], voting='soft')

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)
f1_scores = []

for train_index, val_index in kf.split(train_data):
    X_train, X_val = train_data[train_index], train_data[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Train the ensemble model
    voting_model.fit(X_train, y_train)

    # Validate the model
    y_pred = voting_model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')
    f1_scores.append(f1)

    print(f"Fold F1 Score: {f1:.4f}")

print(f"Mean F1 Score: {np.mean(f1_scores):.4f}")

# Train final model on full training data
voting_model.fit(train_data, y)

# Predict on val/test data
valtest_preds = voting_model.predict(valtest_data)

# Save submission file
file_name = "submission_svm_hyperopt.csv"
file_path = os.path.join(os.getcwd(), file_name)

submission = pd.DataFrame({
    'ID': np.arange(len(valtest_preds)),
    'Predicted': valtest_preds
})
submission.to_csv(file_path, index=False)

print(f"Submission file created: {file_name}")
