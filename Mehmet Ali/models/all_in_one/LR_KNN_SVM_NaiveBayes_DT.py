# importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

# Scale the data
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
valtest_data = scaler.transform(valtest_data)

# Extract labels
labels = train_labels['label'].values

# Creating a list of classification models
List_Classification_Models = [
    ('LR', LogisticRegression(max_iter=500, solver='lbfgs')),
    ('KNN', KNeighborsClassifier()),
    ('SVM', SVC()),
    ('NaiveBayes', GaussianNB()),
    ('DT', DecisionTreeClassifier())
]

# Create empty lists to store cross-validation results and model names
Model_Eval_Score = []
Name_of_model = []

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)

for name, model_detail in List_Classification_Models:
    # Perform cross-validation
    CV_Results = cross_val_score(model_detail, train_data, labels, cv=kf, scoring='f1_macro')
    # Append the results and model names to their respective lists
    Model_Eval_Score.append(CV_Results)
    Name_of_model.append(name)

# Creating a DataFrame with cross-validation results
CV_IterationsBy_model = pd.DataFrame(Model_Eval_Score, index=Name_of_model)

print("The 5 cross-validation results of each classification algorithm are:\n")
Table_Results_CV = CV_IterationsBy_model.T
print(Table_Results_CV)

# Optionally, visualize the cross-validation results
plt.figure(figsize=(10, 6))
plt.boxplot(Model_Eval_Score)
plt.xticks(ticks=range(1, len(Name_of_model)+1), labels=Name_of_model)
plt.title('Cross-Validation F1 Scores by Model')
plt.xlabel('Classification Model')
plt.ylabel('F1 Score')
plt.show()

# Print summary statistics for the cross-validation scores
print("\nSummary statistics for cross-validation results:")
print(Table_Results_CV.describe())