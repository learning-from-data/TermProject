import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline, FeatureUnion  # Import FeatureUnion
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

# Load training data (combine loading steps)
train_data = load_data('train_feats.npy')
train_labels = pd.read_csv('train_labels.csv')['label'].values
valtest_data = load_data('valtest_feats.npy')

# Feature keys
feature_keys = ['resnet_feature', 'vit_feature', 'clip_feature', 'dino_feature']

# Define the SVM pipeline with FeatureUnion
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_union', FeatureUnion(
        transformer_list=[
            ('pca', PCA(n_components=100)),
            ('rbf_sampler', RBFSampler(gamma=1.0, random_state=1))
        ]
    )),
    ('svc', SVC(kernel='linear', random_state=1))
])

# Best parameters initialization
best_f1_score = 0
best_combination = None

# Iterate over feature combinations (using list comprehension for efficiency)
feature_combinations = [combo for n in range(1, len(feature_keys) + 1) for combo in combinations(feature_keys, n)]

for combination in feature_combinations:
    # Select features efficiently using dictionary indexing
    train_subset = np.concatenate([train_data[key] for key in combination], axis=1)
    valtest_subset = np.concatenate([valtest_data[key] for key in combination], axis=1)

    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_subset, train_labels, test_size=0.2, random_state=1, stratify=train_labels
    )

    # Perform grid search on the pipeline
    param_grid = {
        'svc__C': [0.1, 1, 10, 100, 500, 1000],
        'svc__gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
    }

    grid_search = GridSearchCV(
        svm_pipeline,
        param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1
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
        best_params=grid_search.best_params_

print(f"Best Combination: {best_combination}, Best Validation F1 Score: {best_f1_score:.4f}")

# Train final model on full training data with the best feature combination
train_subset = np.concatenate([train_data[key] for key in best_combination], axis=1)
valtest_subset = np.concatenate([valtest_data[key] for key in best_combination], axis=1)

# Train the pipeline on full training data
svm_pipeline.set_params(**best_params) #Use best_params to set parameters
svm_pipeline.fit(train_subset, train_labels)

# Predict on val/test data
valtest_preds = svm_pipeline.predict(valtest_subset)

# Save submission file
file_name = "submission.csv"
file_path = os.path.join(os.getcwd(), file_name)

submission = pd.DataFrame({
    'ID': np.arange(len(valtest_preds)),
    'Predicted': valtest_preds
})
submission.to_csv(file_path, index=False)

print(f"Submission file created: {file_name}")