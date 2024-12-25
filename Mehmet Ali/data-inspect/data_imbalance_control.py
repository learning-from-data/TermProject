import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Load training data
train_feats = np.load('../models/svm/train-files/train_feats.npy', allow_pickle=True).item()
train_labels = pd.read_csv('../models/svm/train-files/train_labels.csv')
valtest_feats = np.load('../models/svm/train-files/valtest_feats.npy', allow_pickle=True).item()

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

# Analyze class distribution
class_distribution = train_labels['label'].value_counts()
print("Class Distribution:")
print(class_distribution)

# Check imbalance ratio
imbalance_ratio = class_distribution.max() / class_distribution.min()
print(f"Imbalance Ratio: {imbalance_ratio}")

# Encode labels
label_encoder = LabelEncoder()
train_labels['label'] = label_encoder.fit_transform(train_labels['label'])

# Compute class weights to handle imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels['label']),
    y=train_labels['label']
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)
f1_scores = []

for fold, (train_index, val_index) in enumerate(kf.split(train_data)):
    print(f"Fold {fold + 1}")

    # Split data
    X_train, X_val = train_data[train_index], train_data[val_index]
    y_train, y_val = train_labels['label'].iloc[train_index], train_labels['label'].iloc[val_index]

    # Train model
    model = RandomForestClassifier(random_state=1, class_weight=class_weight_dict)
    model.fit(X_train, y_train)

    # Validate model
    y_val_preds = model.predict(X_val)
    fold_f1 = f1_score(y_val, y_val_preds, average='weighted')
    f1_scores.append(fold_f1)
    print(f"Fold {fold + 1} F1 Score: {fold_f1}")

# Print average F1 score
average_f1 = np.mean(f1_scores)
print(f"Average F1 Score: {average_f1}")

# Train final model on entire training data
final_model = RandomForestClassifier(random_state=1, class_weight=class_weight_dict)
final_model.fit(train_data, train_labels['label'])

# Predict on validation/test data
valtest_preds = final_model.predict(valtest_data)
valtest_preds = label_encoder.inverse_transform(valtest_preds)

# Save predictions
submission = pd.DataFrame({
    'ID': np.arange(len(valtest_preds)),
    'Predicted': valtest_preds
})
file_name = "submission_imbalance_control.csv"
file_path = os.path.join(os.getcwd(), file_name)
submission.to_csv(file_path, index=False)

print(f"Submission file created: {file_path}")
