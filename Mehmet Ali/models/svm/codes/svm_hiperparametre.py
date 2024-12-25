import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import optuna
import os

# Load training data
train_feats = np.load('../train-files/train_feats.npy', allow_pickle=True).item()
train_labels = pd.read_csv('../train-files/train_labels.csv')
valtest_feats = np.load('../train-files/valtest_feats.npy', allow_pickle=True).item()

# Combine feature vectors into a single feature matrix for training and testing
train_data = np.hstack([
    train_feats['clip_feature'],
    train_feats['dino_feature']
])

valtest_data = np.hstack([
    valtest_feats['clip_feature'],
    valtest_feats['dino_feature']
])

# Labels
y = train_labels['label'].values

# Normalize the features
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
valtest_data = scaler.transform(valtest_data)

# Define Bayesian Optimization objective function
def objective(trial):
    # Suggest hyperparameters
    C = trial.suggest_float('C', 0.1, 100, log=True)
    gamma = trial.suggest_float('gamma', 0.001, 1, log=True)
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])

    # Create the model
    model = SVC(kernel='rbf', C=C, gamma=gamma, class_weight=class_weight, random_state=1)

    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    f1_scores = []

    for train_index, val_index in kf.split(train_data):
        X_train, X_val = train_data[train_index], train_data[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Train the model
        model.fit(X_train, y_train)

        # Validate the model
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='macro')
        f1_scores.append(f1)

    # Return the mean F1 score
    return np.mean(f1_scores)

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Print the best parameters and F1 score
print("Best parameters:", study.best_params)
print("Best cross-validation F1 score:", study.best_value)

# Train final model with the best parameters
final_model = SVC(kernel='rbf', C=study.best_params['C'], 
                  gamma=study.best_params['gamma'], 
                  class_weight=study.best_params['class_weight'], 
                  random_state=1)
final_model.fit(train_data, y)

# Predict on val/test data
valtest_preds = final_model.predict(valtest_data)

# Save predictions to a submission file
submission = pd.DataFrame({
    'ID': np.arange(len(valtest_preds)),
    'Predicted': valtest_preds
})
file_name = "submission_svm_bayesian_optimized.csv"
file_path = os.path.join(os.getcwd(), file_name)
submission.to_csv(file_path, index=False)

print(f"Submission file created: {file_name}")
