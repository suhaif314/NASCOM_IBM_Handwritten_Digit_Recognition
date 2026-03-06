"""
Train all models for Lung Cancer Prediction (standalone script).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  LUNG CANCER PREDICTION - ML & DEEP LEARNING")
print("=" * 60)

# Load data
print("\n[1/8] Loading data...")
df = pd.read_csv("lung_cancer_data.csv")
print(f"Shape: {df.shape}")

# Preprocess
print("\n[2/8] Preprocessing...")
le_g = LabelEncoder()
df['GENDER'] = le_g.fit_transform(df['GENDER'])
le_c = LabelEncoder()
df['LUNG_CANCER'] = le_c.fit_transform(df['LUNG_CANCER'])

X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
print("\n[3/8] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ML Models
print("\n[4/8] Training ML models...")
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    results[name] = {'accuracy': acc, 'auc': auc}
    print(f"  {name}: Acc={acc*100:.2f}%, AUC={auc:.4f}")

# ANN
print("\n[5/8] Training ANN...")
ann = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.summary()

history = ann.fit(X_train, y_train, epochs=50, batch_size=32,
                  validation_split=0.2, verbose=1)

ann_loss, ann_acc = ann.evaluate(X_test, y_test, verbose=0)
ann_prob = ann.predict(X_test, verbose=0).flatten()
ann_auc = roc_auc_score(y_test, ann_prob)
results['ANN (Deep Learning)'] = {'accuracy': ann_acc, 'auc': ann_auc}

print(f"\n  ANN: Acc={ann_acc*100:.2f}%, AUC={ann_auc:.4f}")

# Report
print("\n[6/8] Classification Reports...")
ann_pred = (ann_prob >= 0.5).astype(int)
print("\nRandom Forest:")
rf_pred = models['Random Forest'].predict(X_test)
print(classification_report(y_test, rf_pred, target_names=['No Cancer', 'Cancer']))
print("ANN:")
print(classification_report(y_test, ann_pred, target_names=['No Cancer', 'Cancer']))

# Save
print("\n[7/8] Saving models...")
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(models['Random Forest'], f)
ann.save('ann_model.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump({'gender': le_g, 'cancer': le_c}, f)
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

comp = pd.DataFrame([
    {'Model': k, 'Accuracy (%)': v['accuracy']*100, 'AUC-ROC': v['auc']}
    for k, v in results.items()
]).sort_values('Accuracy (%)', ascending=False).reset_index(drop=True)
comp.to_csv('model_comparison.csv', index=False)

print("\n[8/8] Done!")
print("\n" + "=" * 60)
print("  MODEL COMPARISON")
print("=" * 60)
print(comp.to_string(index=False))
print("=" * 60)
best = comp.iloc[0]
print(f"\nBest Model: {best['Model']} ({best['Accuracy (%)']:.2f}%)")
