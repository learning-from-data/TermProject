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
from sklearn.preprocessing import PolynomialFeatures
from scipy.sparse import csr_matrix  # Use sparse matrix for efficiency
import os
import time

def load_data(file_path):
    """Load data from a file path with error handling."""
    try:
        return np.load(file_path, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


# Load data
train_data = load_data('train_feats.npy')
train_labels = pd.read_csv('train_labels.csv')['label'].values
valtest_data = load_data('valtest_feats.npy')

# Feature keys
feature_keys = ['resnet_feature', 'vit_feature', 'clip_feature', 'dino_feature']

def apply_pca(X_train, X_val, n_components):
    print("Pca appyling beginning...")
    """Applies PCA separately to training and validation sets."""
    pca = PCA(n_components=n_components, random_state=1)  # Add random_state for reproducibility
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    return X_train_pca, X_val_pca

# Define the SVM pipeline OUTSIDE the loop
svm_pipeline = Pipeline([
    ('feature_union', FeatureUnion(
        transformer_list=[
            ('rbf_sampler', RBFSampler(gamma=1.0, random_state=1)), #Keep the RBF sampler outside as well
        ]
    )),
    ('svc', SVC(random_state=1))
])

best_f1_score = 0
best_combination = None
best_params = None

# Pre-calculate all feature concatenations (major speedup)
all_concatenations = {}
print("Pre-calculating feature concatenations...")
start_time= time.time()
for combination in [combo for n in range(1, len(feature_keys) + 1) for combo in combinations(feature_keys, n)]:
    all_concatenations[combination] = np.concatenate([train_data[key] for key in combination], axis=1)
end_time= time.time()
print(f"Feature concatenation pre-calculation took {end_time - start_time:.2f} seconds.")
for combination, train_subset in all_concatenations.items():
    print("\nProcessing combination.")
    # Split data FIRST
    X_train, X_val, y_train, y_val = train_test_split(
        train_subset, train_labels, test_size=0.2, random_state=1, stratify=train_labels
    )

    # Scale data AFTER splitting
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    print("Scaled")
    # Apply PCA AFTER splitting and scaling
    pca_components = min(50, X_train.shape[1])
    X_train_pca, X_val_pca = apply_pca(X_train, X_val, pca_components)
    print("PCA applied on X train and X val")
    # Perform grid search (use X_train_pca and X_val_pca here)
    param_grid = {
        'svc__C': [0.1, 1, 10, 100, 500, 1000],
        'svc__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
        'svc__kernel': ['linear', 'rbf', 'poly']
    }
    print("Param_grid created.")
    grid_search = GridSearchCV(
        svm_pipeline,
        param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=4  # Use all available cores
    )
    print("GridSearch executed.")
    grid_search.fit(X_train_pca, y_train)
    print("GridSearch FIT executed.")
    y_pred = grid_search.best_estimator_.predict(X_val_pca)
    print("GridSearch BEST ESTIMATOR+PREDICT executed.")
    val_f1_score = f1_score(y_val, y_pred, average='macro')
    print("F1 score must be below:")
    print(f"Combination: {combination}, Best Params: {grid_search.best_params_}, Validation F1 Score: {val_f1_score:.4f}")

    if val_f1_score > best_f1_score:
        best_f1_score = val_f1_score
        best_combination = combination
        best_params = grid_search.best_params_

print(f"Best Combination: {best_combination}, Best Validation F1 Score: {best_f1_score:.4f}")

# Final training (using best combination)
train_subset = all_concatenations[best_combination]
valtest_subset = np.concatenate([valtest_data[key] for key in best_combination], axis=1)

scaler = StandardScaler()
train_subset = scaler.fit_transform(train_subset)
valtest_subset = scaler.transform(valtest_subset)

pca_components = min(50, train_subset.shape[1])
train_subset, valtest_subset = apply_pca(train_subset, valtest_subset, pca_components)

svm_pipeline.set_params(**best_params)
svm_pipeline.fit(train_subset, train_labels)

valtest_preds = svm_pipeline.predict(valtest_subset)

# Save predictions
submission = pd.DataFrame({'ID': np.arange(len(valtest_preds)), 'Predicted': valtest_preds})
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")
