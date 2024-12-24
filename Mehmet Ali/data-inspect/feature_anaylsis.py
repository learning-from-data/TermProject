import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
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

# Korelasyon Analizi
# Özelliklerin isimlerini belirlemek
feature_names = ['resnet_feature', 'vit_feature', 'clip_feature', 'dino_feature']

# train_data ve train_labels'i birleştirerek DataFrame oluşturma
train_df = pd.DataFrame(train_data, columns=[f'{name}_{i}' for name in feature_names for i in range(train_feats[name].shape[1])])
train_df['label'] = train_labels['label']

# Korelasyon matrisi
corr_matrix = train_df.corr()

# Hedef değişken ile korelasyonları seçme
label_correlation = corr_matrix['label'].drop('label')

# Korelasyonu görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Korelasyon Matrisi")
plt.show()

# En yüksek korelasyona sahip özellikleri sıralama
top_features = label_correlation.abs().sort_values(ascending=False).head(10)
print("Hedef değişkenle en yüksek korelasyona sahip özellikler:")
print(top_features)

# Özellik Önem Değerlerini Belirleme
# Veri standardizasyonu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_data)

# Lojistik Regresyon modeli eğitimi
model = LogisticRegression()
model.fit(X_scaled, train_labels['label'])

# Özellik ağırlıkları (önem değerleri)
feature_importances = model.coef_[0]

# Ağırlıkları bir tabloya dönüştürme
importance_df = pd.DataFrame({
    'Feature': [f'{name}_{i}' for name in feature_names for i in range(train_feats[name].shape[1])],
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Özellik Önem Değerleri:")
print(importance_df.head(10))

# Görselleştirme
plt.figure(figsize=(10, 6))
importance_df.head(10).plot(kind='bar', x='Feature', y='Importance', legend=False)
plt.title("En Önemli Özellikler")
plt.ylabel("Önem Değeri")
plt.xlabel("Özellikler")
plt.show()

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)
f1_scores = []

for train_index, val_index in kf.split(train_data):
    X_train, X_val = train_data[train_index], train_data[val_index]
    y_train, y_val = train_labels.iloc[train_index]['label'], train_labels.iloc[val_index]['label']
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    f1 = f1_score(y_val, y_pred, average='macro')
    f1_scores.append(f1)

print("5-Fold Cross-Validation F1 Scores:", f1_scores)
print("Average F1 Score:", np.mean(f1_scores))

# Modeli validation-test veri setiyle değerlendirme
valtest_preds = model.predict(valtest_data)

# Submission dosyası oluşturma
file_name = "submission_logistic_regression.csv"
file_path = os.path.join(os.getcwd(), file_name)

submission = pd.DataFrame({
    'ID': np.arange(len(valtest_preds)),
    'Predicted': valtest_preds
})
submission.to_csv(file_path, index=False)

print(f"Submission file created: {file_name}")
