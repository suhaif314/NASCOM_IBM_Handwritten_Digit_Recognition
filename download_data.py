"""
Download / Generate Lung Cancer Survey Dataset
Based on the well-known lung cancer patient survey dataset.
Features represent patient symptoms and habits.
Target: LUNG_CANCER (YES / NO)
"""

import numpy as np
import pandas as pd

np.random.seed(42)

n_samples = 1000

# --- Generate features with realistic medical correlations ---
age = np.random.randint(20, 80, n_samples)
gender = np.random.choice(['M', 'F'], n_samples, p=[0.55, 0.45])

# Smoking correlates with age (older more likely)
smoking_prob = 0.3 + (age - 20) / 120
smoking = np.array([np.random.binomial(1, min(p, 0.85)) for p in smoking_prob])

# Yellow fingers strongly correlated with smoking
yellow_fingers = np.array([np.random.binomial(1, 0.7 if s else 0.1) for s in smoking])

# Anxiety - somewhat random
anxiety = np.random.binomial(1, 0.45, n_samples)

# Peer pressure
peer_pressure = np.random.binomial(1, 0.40, n_samples)

# Chronic disease correlates with age
chronic_prob = 0.1 + (age - 20) / 100
chronic_disease = np.array([np.random.binomial(1, min(p, 0.8)) for p in chronic_prob])

# Fatigue correlates with chronic disease and age
fatigue = np.array([np.random.binomial(1, 0.7 if cd else 0.3) for cd in chronic_disease])

# Allergy
allergy = np.random.binomial(1, 0.45, n_samples)

# Wheezing correlates with smoking
wheezing = np.array([np.random.binomial(1, 0.65 if s else 0.15) for s in smoking])

# Alcohol consuming
alcohol = np.random.binomial(1, 0.50, n_samples)

# Coughing correlates with smoking
coughing = np.array([np.random.binomial(1, 0.75 if s else 0.2) for s in smoking])

# Shortness of breath correlates with smoking and chronic disease
sob = np.array([np.random.binomial(1, 0.8 if (s and cd) else (0.5 if s else (0.3 if cd else 0.1)))
                for s, cd in zip(smoking, chronic_disease)])

# Swallowing difficulty
swallowing_diff = np.random.binomial(1, 0.30, n_samples)

# Chest pain correlates with smoking and chronic disease
chest_pain = np.array([np.random.binomial(1, 0.6 if s else 0.2) for s in smoking])

# --- Generate target (LUNG_CANCER) based on risk factors ---
# Higher risk: smoking, age>50, chronic disease, coughing, shortness of breath
risk_score = (
    smoking * 3.0 +
    (age > 50).astype(int) * 2.0 +
    chronic_disease * 1.5 +
    coughing * 1.2 +
    sob * 1.5 +
    wheezing * 1.0 +
    yellow_fingers * 0.8 +
    chest_pain * 1.0 +
    fatigue * 0.5 +
    alcohol * 0.5 +
    swallowing_diff * 0.7 +
    (age > 60).astype(int) * 1.0
)

# Convert risk score to probability using sigmoid-like function
risk_normalized = (risk_score - risk_score.mean()) / risk_score.std()
cancer_prob = 1 / (1 + np.exp(-risk_normalized * 1.5))
lung_cancer = np.array([np.random.binomial(1, p) for p in cancer_prob])

# Encode: 2=YES, 1=NO for symptom columns (matching original dataset format)
# But we'll use cleaner 1/0 encoding
gender_encoded = np.where(np.array(gender) == 'M', 1, 0)

# Create DataFrame
df = pd.DataFrame({
    'GENDER': gender,
    'AGE': age,
    'SMOKING': smoking,
    'YELLOW_FINGERS': yellow_fingers,
    'ANXIETY': anxiety,
    'PEER_PRESSURE': peer_pressure,
    'CHRONIC_DISEASE': chronic_disease,
    'FATIGUE': fatigue,
    'ALLERGY': allergy,
    'WHEEZING': wheezing,
    'ALCOHOL_CONSUMING': alcohol,
    'COUGHING': coughing,
    'SHORTNESS_OF_BREATH': sob,
    'SWALLOWING_DIFFICULTY': swallowing_diff,
    'CHEST_PAIN': chest_pain,
    'LUNG_CANCER': np.where(lung_cancer == 1, 'YES', 'NO')
})

# Save
df.to_csv('lung_cancer_data.csv', index=False)
print(f"Dataset saved: lung_cancer_data.csv")
print(f"Shape: {df.shape}")
print(f"\nTarget Distribution:")
print(df['LUNG_CANCER'].value_counts())
print(f"\nFeature Summary:")
print(df.describe())
print(f"\nFirst 5 rows:")
print(df.head())
