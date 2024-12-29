import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Best parameters initialization
best_f1_score = 0
best_combination = None

# Define the SVM pipeline
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=100)),  # Adjusted PCA components dynamically
    ('rbf_sampler', RBFSampler(gamma=1.0, random_state=1)),
    ('svc', SVC(kernel='linear', random_state=1))  # Linear kernel with RBF approximation
])

# Iterate over feature combinations
for n in range(1, len(feature_keys) + 1):
    for combination in combinations(feature_keys, n):
        # Combine selected features
        train_data = np.hstack([train_feats[key] for key in combination])
        valtest_data = np.hstack([valtest_feats[key] for key in combination])

        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            train_data, y, test_size=0.2, random_state=1, stratify=y
        )

        # Perform grid search on the pipeline
        param_grid = {
            'svc__C': [0.1, 1, 10, 100, 500, 1000],
            'svc__gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
        }

        grid_search = GridSearchCV(
            svm_pipeline,
            param_grid,
            cv=5,  # 5-fold cross-validation
            scoring='f1_macro',
            n_jobs=-1  # Use all available processors
        )

        grid_search.fit(X_train, y_train)

        # Validate the model
        y_pred = grid_search.best_estimator_.predict(X_val)
        val_f1_score = f1_score(y_val, y_pred, average='macro')

        print(f"Combination: {combination}, Best Params: {grid_search.best_params_}, Validation F1 Score: {val_f1_score:.4f}")

        # Update best combination if necessary
        if val_f1_score > best_f1_score:
            best_f1_score = val_f1_score
            best_combination = combination

print(f"Best Combination: {best_combination}, Best Validation F1 Score: {best_f1_score:.4f}")

# Train final model on full training data with the best feature combination
train_data = np.hstack([train_feats[key] for key in best_combination])
valtest_data = np.hstack([valtest_feats[key] for key in best_combination])

# Train the pipeline on full training data
svm_pipeline.set_params(
    svc__C=grid_search.best_params_['svc__C'],
    svc__gamma=grid_search.best_params_['svc__gamma']
)

svm_pipeline.fit(train_data, y)

# Predict on val/test data
valtest_preds = svm_pipeline.predict(valtest_data)

# Save submission file
file_name = "submission.csv"
file_path = os.path.join(os.getcwd(), file_name)

submission = pd.DataFrame({
    'ID': np.arange(len(valtest_preds)),
    'Predicted': valtest_preds
})
submission.to_csv(file_path, index=False)

print(f"Submission file created: {file_name}")
