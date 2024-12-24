import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

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

# Initialize Random Forest
clf = RandomForestClassifier(random_state=42, n_estimators=100)

# Perform 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)
f1_scores = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')
    f1_scores.append(f1)

average_f1_score = np.mean(f1_scores)
print(f"Average F1 Score from 5-Fold CV: {average_f1_score}")

# Load test data
valtest_feats = np.load('valtest_feats.npy', allow_pickle=True).item()
X_test = np.hstack([
    valtest_feats['clip_feature'],
    valtest_feats['dino_feature'],
    valtest_feats['resnet_feature'],
    valtest_feats['vit_feature']
])

# Train on full dataset and predict test set
clf.fit(X, y)
predictions = clf.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'ID': range(len(predictions)),
    'Predicted': predictions
})

# Ensure the submission matches the required format
submission.to_csv('submission-kfold.csv', index=False)

print("Submission file 'submission-kfold.csv' created.")
