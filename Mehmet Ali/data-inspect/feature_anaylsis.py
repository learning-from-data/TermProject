import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
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

# En yüksek korelasyona sahip özellikleri sıralama
top_features = label_correlation.abs().sort_values(ascending=False).head(100)
print("Hedef değişkenle en yüksek korelasyona sahip özellikler:")
print(top_features)

# En önemli özelliklere göre veri alt kümesi oluşturma
selected_features = top_features.index.tolist()
train_data = train_df[selected_features].values

# Validation-test verisi için aynı seçimi uygulama
valtest_df = pd.DataFrame(valtest_data, columns=[f'{name}_{i}' for name in feature_names for i in range(valtest_feats[name].shape[1])])
valtest_data = valtest_df[selected_features].values

# Veri standardizasyonu
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
valtest_data = scaler.transform(valtest_data)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)
f1_scores = []

for train_index, val_index in kf.split(train_data):
    X_train, X_val = train_data[train_index], train_data[val_index]
    y_train, y_val = train_labels.iloc[train_index]['label'], train_labels.iloc[val_index]['label']
    
    model = SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    f1 = f1_score(y_val, y_pred, average='macro')
    f1_scores.append(f1)

print("5-Fold Cross-Validation F1 Scores:", f1_scores)
print("Average F1 Score:", np.mean(f1_scores))

# Modeli validation-test veri setiyle değerlendirme
model.fit(train_data, train_labels['label'])
valtest_preds = model.predict(valtest_data)

# Submission dosyası oluşturma
file_name = "submission_svm_w_best_features.csv"
file_path = os.path.join(os.getcwd(), file_name)

submission = pd.DataFrame({
    'ID': np.arange(len(valtest_preds)),
    'Predicted': valtest_preds
})
submission.to_csv(file_path, index=False)

print(f"Submission file created: {file_name}")
