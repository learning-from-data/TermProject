import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import os

# This uses gpu

# Load training data
train_feats = np.load('/kaggle/input/lfd-project-dataset/train_feats.npy', allow_pickle=True).item()
train_labels = pd.read_csv('/kaggle/input/lfd-project-dataset/train_labels.csv')
valtest_feats = np.load('/kaggle/input/lfd-project-dataset/valtest_feats.npy', allow_pickle=True).item()

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

# Labels
labels = train_labels['label'].values

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.softmax(x)

# Hyperparameters
input_size = train_data.shape[1]
hidden_size = 128
output_size = len(np.unique(labels))
epochs = 20
batch_size = 32
learning_rate = 0.001

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)
f1_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
    print(f"Starting Fold {fold+1}")

    X_train, X_val = train_data[train_idx], train_data[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

    # Model, loss, and optimizer
    model = MLP(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_preds = torch.argmax(val_outputs, axis=1)
        f1 = f1_score(y_val_tensor.cpu().numpy(), val_preds.cpu().numpy(), average='macro')
        f1_scores.append(f1)
        print(f"Fold {fold+1} F1 Score: {f1:.4f}")

# Train on full dataset and predict for validation/test set
X_train_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(labels, dtype=torch.long).to(device)
X_valtest_tensor = torch.tensor(valtest_data, dtype=torch.float32).to(device)

model = MLP(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Full training loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# Predict for val/test data
model.eval()
with torch.no_grad():
    valtest_outputs = model(X_valtest_tensor)
    valtest_preds = torch.argmax(valtest_outputs, axis=1)

# Save predictions
submission = pd.DataFrame({
    'ID': np.arange(len(valtest_preds)),
    'Predicted': valtest_preds.cpu().numpy()
})

file_name = "submission_of_mlp.csv"
file_path = os.path.join(os.getcwd(), file_name)
submission.to_csv(file_path, index=False)

print(f"Submission file created: {file_name}")
