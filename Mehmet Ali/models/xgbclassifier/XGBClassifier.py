import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import f1_score
from sklearn.model_selection import KFold
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

# Extract labels
labels = train_labels['label']

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)
f1_scores = []

# Initialize predictions for valtest data
valtest_preds = np.zeros(valtest_data.shape[0], dtype=int)

# Train and validate the model
for train_index, val_index in kf.split(train_data):
    X_train, X_val = train_data[train_index], train_data[val_index]
    y_train, y_val = labels[train_index], labels[val_index]

    # Initialize XGBClassifier with cpu support
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predict on validation set
    val_preds = model.predict(X_val)

    # Calculate F1 score
    f1 = f1_score(y_val, val_preds, average='weighted')
    f1_scores.append(f1)

    print(f"Fold F1 Score: {f1}")

# Train on the full dataset and predict on valtest data
final_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
final_model.fit(train_data, labels)
valtest_preds = final_model.predict(valtest_data)

# Create submission file
file_name = "submission.csv"
file_path = os.path.join(os.getcwd(), file_name)

submission = pd.DataFrame({
    'ID': np.arange(len(valtest_preds)),
    'Predicted': valtest_preds
})
submission.to_csv(file_path, index=False)

print(f"Submission file created: {file_name}")
print(f"Mean F1 Score: {np.mean(f1_scores)}")
