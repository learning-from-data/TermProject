import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import os
import time

def load_data(file_path):
    """Load data from a file path with error handling."""
    try:
        return np.load(file_path, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def apply_pca(X_train, X_val, variance_threshold=0.95):
    """Applies PCA, retaining components to explain the desired variance."""
    print("Applying PCA...")
    pca = PCA(n_components=variance_threshold, random_state=1)  # Retain variance threshold
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    print(f"PCA applied. Components retained: {pca.n_components_}")
    return X_train_pca, X_val_pca

# Load data
train_data = load_data('train_feats.npy')
train_labels = pd.read_csv('train_labels.csv')['label'].values
valtest_data = load_data('valtest_feats.npy')

if train_data is None or valtest_data is None:
    raise ValueError("Missing input files. Ensure 'train_feats.npy' and 'valtest_feats.npy' are present.")

# Feature keys
feature_keys = ['resnet_feature', 'vit_feature', 'clip_feature', 'dino_feature']

# Pre-calculate all feature concatenations
print("Pre-calculating feature concatenations...")
start_time = time.time()
all_concatenations = {
    combination: np.concatenate([train_data[key] for key in combination], axis=1)
    for n in range(1, len(feature_keys) + 1)
    for combination in combinations(feature_keys, n)
}
end_time = time.time()
print(f"Feature concatenation pre-calculation took {end_time - start_time:.2f} seconds.")

# Initialize SVM pipeline
svm_pipeline = Pipeline([
    ('feature_union', FeatureUnion(
        transformer_list=[
            ('rbf_sampler', RBFSampler(gamma=1.0, random_state=1)),
        ]
    )),
    ('svc', SVC(random_state=1))
])

# Track best results
best_f1_score = 0
best_combination = None
best_params = None

# Main loop
for combination, train_subset in all_concatenations.items():
    print(f"\nProcessing combination: {combination}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        train_subset, train_labels, test_size=0.2, random_state=1, stratify=train_labels
    )

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Apply PCA
    X_train_pca, X_val_pca = apply_pca(X_train, X_val)

    # Grid Search
    param_grid = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 0.1, 0.01],
        'svc__kernel': ['rbf']
    }
    grid_search = GridSearchCV(
        svm_pipeline,
        param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=4,
        verbose=2 # TO follow grid size and fit nmbers.
    )
    print("Starting Grid Search...")
    try:
        grid_search.fit(X_train_pca, y_train)
        y_pred = grid_search.best_estimator_.predict(X_val_pca)
        val_f1_score = f1_score(y_val, y_pred, average='macro')
        print(f"Combination: {combination}, Best Params: {grid_search.best_params_}, Validation F1 Score: {val_f1_score:.4f}")

        # Update best results
        if val_f1_score > best_f1_score:
            best_f1_score = val_f1_score
            best_combination = combination
            best_params = grid_search.best_params_

    except Exception as e:
        print(f"Error during Grid Search for combination {combination}: {e}")

print(f"\nBest Combination: {best_combination}, Best Validation F1 Score: {best_f1_score:.4f}")

# Final training
train_subset = all_concatenations[best_combination]
valtest_subset = np.concatenate([valtest_data[key] for key in best_combination], axis=1)

scaler = StandardScaler()
train_subset = scaler.fit_transform(train_subset)
valtest_subset = scaler.transform(valtest_subset)

train_subset, valtest_subset = apply_pca(train_subset, valtest_subset)

svm_pipeline.set_params(**best_params)
svm_pipeline.fit(train_subset, train_labels)

valtest_preds = svm_pipeline.predict(valtest_subset)

# Save predictions
submission = pd.DataFrame({'ID': np.arange(len(valtest_preds)), 'Predicted': valtest_preds})
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")
