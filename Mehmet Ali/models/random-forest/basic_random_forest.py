import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Load data
train_feats = np.load('train_feats.npy', allow_pickle=True).item()
train_labels = pd.read_csv('train_labels.csv')
valtest_feats = np.load('valtest_feats.npy', allow_pickle=True).item()

# Combine features (CLIP, DINO, ResNet, ViT) into a single matrix
X_train = np.hstack([
    train_feats['clip_feature'],
    train_feats['dino_feature'],
    train_feats['resnet_feature'],
    train_feats['vit_feature']
])
y_train = train_labels['label']

X_test = np.hstack([
    valtest_feats['clip_feature'],
    valtest_feats['dino_feature'],
    valtest_feats['resnet_feature'],
    valtest_feats['vit_feature']
])

# Initialize Random Forest
clf = RandomForestClassifier(random_state=42, n_estimators=100)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'ID': range(len(predictions)),
    'Predicted': predictions
})

# Ensure the submission matches the required format
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created.")
