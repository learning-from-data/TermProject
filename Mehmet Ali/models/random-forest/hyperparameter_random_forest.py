import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, GridSearchCV

# Load data
train_feats = np.load('train_feats.npy', allow_pickle=True).item()
train_labels = pd.read_csv('train_labels.csv')

# Combine features (CLIP, DINO, ResNet, ViT) into a single matrix
X = np.hstack([
    train_feats['clip_feature'],
    train_feats['dino_feature'],
    train_feats['resnet_feature'],
    train_feats['vit_feature']
])
y = train_labels['label']

# Define parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Random Forest
clf = RandomForestClassifier(random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='f1_macro', cv=5, n_jobs=-1, verbose=2, return_train_score=True)
grid_search.fit(X, y)

# Best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"Best Parameters: {best_params}")
print(f"Best F1 Score from Grid Search: {best_score}")

# Load test data
valtest_feats = np.load('valtest_feats.npy', allow_pickle=True).item()
X_test = np.hstack([
    valtest_feats['clip_feature'],
    valtest_feats['dino_feature'],
    valtest_feats['resnet_feature'],
    valtest_feats['vit_feature']
])

# Train on full dataset with best parameters and predict test set
final_clf = RandomForestClassifier(random_state=42, **best_params)
final_clf.fit(X, y)
predictions = final_clf.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'ID': range(len(predictions)),
    'Predicted': predictions
})

# Ensure the submission matches the required format
submission.to_csv('submission-hyparameter.csv', index=False)

print("Submission file 'submission-hyperparameter.csv' created.")
