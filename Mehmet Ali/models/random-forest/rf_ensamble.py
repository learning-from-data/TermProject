import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from joblib import dump

# Load train_feats and valtest_feats
train_feats = np.load('train_feats.npy', allow_pickle=True).item()
valtest_feats = np.load('valtest_feats.npy', allow_pickle=True).item()

# Extract features
X_train_resnet = train_feats['resnet_feature']
X_train_vit = train_feats['vit_feature']
X_train_clip = train_feats['clip_feature']
X_train_dino = train_feats['dino_feature']

# Combine all features
X_train_combined = np.concatenate(
    [X_train_resnet, X_train_vit, X_train_clip, X_train_dino], axis=1
)

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

# Train-validation split
X_train, X_val, y_train_split, y_val_split = train_test_split(
    X_train_combined, y_train, test_size=0.2, random_state=42
)

# Step 1: Train Baseline Random Forest Model
print("Training Baseline Random Forest Model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train_split)

# Validate the baseline model
y_val_pred = rf_model.predict(X_val)
print("Baseline Validation Results:\n", classification_report(y_val_split, y_val_pred))

# Predict on test data
y_test_pred = rf_model.predict(X_test_combined)

# Generate submission file for baseline model
submission = pd.DataFrame({
    'ID': valtest_feats['idx'],
    'label': y_test_pred
})
submission.to_csv('submission_baseline.csv', index=False)
print("Baseline submission file 'submission_baseline.csv' created!")

# Step 2: Hyperparameter Tuning for Random Forest
print("\nStarting Hyperparameter Tuning for Random Forest...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', None]
}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=2,
    n_jobs=-1
)
grid_search.fit(X_train, y_train_split)
best_rf_model = grid_search.best_estimator_

print("Best Parameters for Random Forest:", grid_search.best_params_)
print("Best Score during CV:", grid_search.best_score_)

# Validate the tuned Random Forest model
y_val_pred = best_rf_model.predict(X_val)
print("Tuned Random Forest Validation Results:\n", classification_report(y_val_split, y_val_pred))

# Predict on test data
y_test_pred = best_rf_model.predict(X_test_combined)

# Generate submission file for tuned Random Forest
submission = pd.DataFrame({
    'ID': valtest_feats['idx'],
    'label': y_test_pred
})
submission.to_csv('submission_rf_tuned.csv', index=False)
print("Tuned Random Forest submission file 'submission_rf_tuned.csv' created!")

# Step 3: Train XGBoost Model
print("\nTraining XGBoost Model...")
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    tree_method='gpu_hist',  # Use GPU acceleration
    random_state=42
)
xgb_model.fit(X_train, y_train_split)

# Validate the XGBoost model
y_val_pred = xgb_model.predict(X_val)
print("XGBoost Validation Results:\n", classification_report(y_val_split, y_val_pred))

# Predict on test data
y_test_pred = xgb_model.predict(X_test_combined)

# Generate submission file for XGBoost
submission = pd.DataFrame({
    'ID': valtest_feats['idx'],
    'label': y_test_pred
})
submission.to_csv('submission_xgb.csv', index=False)
print("XGBoost submission file 'submission_xgb.csv' created!")

# Step 4: Train Ensemble Model (Random Forest + XGBoost)
print("\nTraining Ensemble Model (Random Forest + XGBoost)...")
ensemble_model = VotingClassifier(
    estimators=[('rf', best_rf_model), ('xgb', xgb_model)],
    voting='soft'
)
ensemble_model.fit(X_train, y_train_split)

# Validate the ensemble model
y_val_pred = ensemble_model.predict(X_val)
print("Ensemble Model Validation Results:\n", classification_report(y_val_split, y_val_pred))

# Predict on test data
y_test_pred = ensemble_model.predict(X_test_combined)

# Generate submission file for Ensemble Model
submission = pd.DataFrame({
    'ID': valtest_feats['idx'],
    'label': y_test_pred
})
submission.to_csv('submission_ensemble.csv', index=False)
print("Ensemble submission file 'submission_ensemble.csv' created!")

