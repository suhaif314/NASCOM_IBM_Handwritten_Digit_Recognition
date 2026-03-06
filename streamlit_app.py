"""
Lung Cancer Risk Prediction - Streamlit GUI
Project by: N. Mohammed Sohaib (74)
PBEL NASCOM Internship

Run: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lung Cancer Prediction",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 Lung Cancer Risk Prediction")
st.markdown("**Project by: N. Mohammed Sohaib (74) | PBEL NASCOM Internship**")
st.markdown("---")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Dataset Explorer",
    "🧠 Train Models",
    "🔍 Predict New Patient",
    "📈 Model Evaluation"
])


# ─── Helper Functions ────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    """Load the lung cancer dataset."""
    df = pd.read_csv("lung_cancer_data.csv")
    return df


def load_artifacts():
    """Load saved model artifacts."""
    artifacts = {}
    if os.path.exists("random_forest_model.pkl"):
        with open("random_forest_model.pkl", "rb") as f:
            artifacts['rf_model'] = pickle.load(f)
    if os.path.exists("scaler.pkl"):
        with open("scaler.pkl", "rb") as f:
            artifacts['scaler'] = pickle.load(f)
    if os.path.exists("model_comparison.csv"):
        artifacts['comparison'] = pd.read_csv("model_comparison.csv")
    if os.path.exists("training_history.pkl"):
        with open("training_history.pkl", "rb") as f:
            artifacts['history'] = pickle.load(f)
    return artifacts


# ─── Load Data ────────────────────────────────────────────────────────────────
df = load_data()
artifacts = load_artifacts()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Dataset Explorer
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dataset Explorer":
    st.header("📊 Dataset Explorer")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", f"{len(df):,}")
    col2.metric("Features", "15")
    cancer_pct = (df['LUNG_CANCER'] == 'YES').sum() / len(df) * 100
    col3.metric("Cancer Cases", f"{(df['LUNG_CANCER'] == 'YES').sum()}")
    col4.metric("Cancer Rate", f"{cancer_pct:.1f}%")

    # Target distribution
    st.subheader("Target Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ['#2ecc71', '#e74c3c']
    df['LUNG_CANCER'].value_counts().plot(kind='bar', ax=axes[0], color=colors)
    axes[0].set_title('Lung Cancer Distribution')
    axes[0].set_xlabel('Lung Cancer')
    axes[0].set_ylabel('Count')
    axes[0].set_xticklabels(['NO', 'YES'], rotation=0)

    df['LUNG_CANCER'].value_counts().plot(kind='pie', ax=axes[1],
        labels=['NO', 'YES'], autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1].set_title('Proportion')
    axes[1].set_ylabel('')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Age distribution
    st.subheader("Age Distribution by Cancer Status")
    fig, ax = plt.subplots(figsize=(10, 4))
    df[df['LUNG_CANCER'] == 'NO']['AGE'].hist(alpha=0.6, bins=20, label='No Cancer', color='green', ax=ax)
    df[df['LUNG_CANCER'] == 'YES']['AGE'].hist(alpha=0.6, bins=20, label='Cancer', color='red', ax=ax)
    ax.set_title('Age Distribution')
    ax.set_xlabel('Age')
    ax.legend()
    st.pyplot(fig)
    plt.close()

    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    df_encoded = df.copy()
    df_encoded['GENDER'] = df_encoded['GENDER'].map({'M': 1, 'F': 0})
    df_encoded['LUNG_CANCER'] = df_encoded['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

    fig, ax = plt.subplots(figsize=(12, 8))
    corr = df_encoded.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    st.pyplot(fig)
    plt.close()

    # Correlation with target
    st.subheader("Correlation with Lung Cancer")
    target_corr = corr['LUNG_CANCER'].drop('LUNG_CANCER').sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = ['green' if x > 0 else 'red' for x in target_corr.values]
    target_corr.plot(kind='barh', color=bar_colors, ax=ax)
    ax.set_title('Feature Correlation with Lung Cancer')
    ax.axvline(x=0, color='black', linestyle='--')
    st.pyplot(fig)
    plt.close()

    # Raw data
    st.subheader("Raw Data")
    st.dataframe(df.head(30))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Train Models
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Train Models":
    st.header("🧠 Model Training")
    st.markdown("Train multiple ML models and an ANN for lung cancer prediction.")

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier

    epochs = st.slider("ANN Epochs", 10, 100, 50)

    if st.button("🚀 Train All Models", type="primary"):
        # Prepare data
        df_train = df.copy()
        le_g = LabelEncoder()
        df_train['GENDER'] = le_g.fit_transform(df_train['GENDER'])
        le_c = LabelEncoder()
        df_train['LUNG_CANCER'] = le_c.fit_transform(df_train['LUNG_CANCER'])

        X = df_train.drop('LUNG_CANCER', axis=1)
        y = df_train['LUNG_CANCER']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y)

        # ML Models
        ml_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }

        results = {}
        progress = st.progress(0)

        for i, (name, model) in enumerate(ml_models.items()):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            results[name] = {'accuracy': acc, 'auc': auc}
            progress.progress((i + 1) / 6)
            st.text(f"✅ {name}: Accuracy={acc*100:.2f}%, AUC={auc:.4f}")

        # ANN
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout

        ann = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        history = ann.fit(X_train, y_train, epochs=epochs, batch_size=32,
                          validation_split=0.2, verbose=0)

        ann_loss, ann_acc = ann.evaluate(X_test, y_test, verbose=0)
        ann_prob = ann.predict(X_test, verbose=0).flatten()
        ann_auc = roc_auc_score(y_test, ann_prob)
        results['ANN'] = {'accuracy': ann_acc, 'auc': ann_auc}
        progress.progress(1.0)
        st.text(f"✅ ANN: Accuracy={ann_acc*100:.2f}%, AUC={ann_auc:.4f}")

        # Save everything
        with open('random_forest_model.pkl', 'wb') as f:
            pickle.dump(ml_models['Random Forest'], f)
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

        st.success("✅ All models trained and saved!")
        st.dataframe(comp)

        # Training curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('ANN Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.plot(history.history['loss'], label='Train')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('ANN Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Predict New Patient
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict New Patient":
    st.header("🔍 Predict Lung Cancer Risk for a New Patient")

    if 'rf_model' not in artifacts or 'scaler' not in artifacts:
        st.warning("⚠️ No trained model found. Please train models first on the 'Train Models' page.")
    else:
        st.markdown("### Enter Patient Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 20, 80, 50)
            smoking = st.selectbox("Smoking", ["No", "Yes"])
            yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"])
            anxiety = st.selectbox("Anxiety", ["No", "Yes"])

        with col2:
            peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"])
            chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"])
            fatigue = st.selectbox("Fatigue", ["No", "Yes"])
            allergy = st.selectbox("Allergy", ["No", "Yes"])
            wheezing = st.selectbox("Wheezing", ["No", "Yes"])

        with col3:
            alcohol = st.selectbox("Alcohol Consuming", ["No", "Yes"])
            coughing = st.selectbox("Coughing", ["No", "Yes"])
            shortness = st.selectbox("Shortness of Breath", ["No", "Yes"])
            swallowing = st.selectbox("Swallowing Difficulty", ["No", "Yes"])
            chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])

        if st.button("🩺 Predict Risk", type="primary"):
            # Encode inputs
            binary = lambda x: 1 if x == "Yes" else 0
            input_data = np.array([[
                1 if gender == "Male" else 0,
                age,
                binary(smoking),
                binary(yellow_fingers),
                binary(anxiety),
                binary(peer_pressure),
                binary(chronic_disease),
                binary(fatigue),
                binary(allergy),
                binary(wheezing),
                binary(alcohol),
                binary(coughing),
                binary(shortness),
                binary(swallowing),
                binary(chest_pain)
            ]])

            # Scale
            input_scaled = artifacts['scaler'].transform(input_data)

            # Predict with Random Forest
            rf_pred = artifacts['rf_model'].predict(input_scaled)[0]
            rf_prob = artifacts['rf_model'].predict_proba(input_scaled)[0]

            st.markdown("---")
            st.markdown("### 📋 Prediction Results")

            col1, col2 = st.columns(2)

            with col1:
                if rf_pred == 1:
                    st.error(f"⚠️ **HIGH RISK** of Lung Cancer")
                else:
                    st.success(f"✅ **LOW RISK** of Lung Cancer")

                st.metric("Cancer Probability", f"{rf_prob[1]*100:.1f}%")
                st.metric("No Cancer Probability", f"{rf_prob[0]*100:.1f}%")

            with col2:
                fig, ax = plt.subplots(figsize=(5, 3))
                bars = ax.bar(['No Cancer', 'Cancer'], rf_prob * 100,
                             color=['#2ecc71', '#e74c3c'])
                ax.set_ylabel('Probability (%)')
                ax.set_title('Risk Assessment')
                ax.set_ylim(0, 105)
                for bar, prob in zip(bars, rf_prob):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{prob*100:.1f}%', ha='center', fontsize=12)
                st.pyplot(fig)
                plt.close()

            # Risk factors summary
            st.markdown("### 🔎 Patient Risk Factors")
            risk_factors = []
            if smoking == "Yes": risk_factors.append("🚬 Smoker")
            if age > 55: risk_factors.append("👴 Age > 55")
            if chronic_disease == "Yes": risk_factors.append("🏥 Chronic Disease")
            if coughing == "Yes": risk_factors.append("😷 Persistent Coughing")
            if shortness == "Yes": risk_factors.append("😮‍💨 Shortness of Breath")
            if chest_pain == "Yes": risk_factors.append("💔 Chest Pain")
            if wheezing == "Yes": risk_factors.append("🌬️ Wheezing")

            if risk_factors:
                for rf in risk_factors:
                    st.markdown(f"- {rf}")
            else:
                st.markdown("No major risk factors detected.")

            st.markdown("---")
            st.caption("⚠️ This is a machine learning model for educational purposes only. "
                      "Always consult a medical professional for actual health concerns.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Model Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Evaluation":
    st.header("📈 Model Evaluation & Comparison")

    if 'comparison' not in artifacts:
        st.warning("⚠️ No trained models found. Please train models first.")
    else:
        comp = artifacts['comparison']

        st.subheader("Model Comparison Table")
        st.dataframe(comp, use_container_width=True)

        best = comp.iloc[0]
        st.success(f"🏆 Best Model: **{best['Model']}** with **{best['Accuracy (%)']:.2f}%** accuracy")

        # Bar charts
        st.subheader("Accuracy & AUC Comparison")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        colors_bar = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
        bars = ax1.bar(comp['Model'], comp['Accuracy (%)'], color=colors_bar[:len(comp)])
        ax1.set_title('Accuracy Comparison')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim(0, 105)
        for bar, acc in zip(bars, comp['Accuracy (%)']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{acc:.1f}%', ha='center', fontsize=9)
        ax1.tick_params(axis='x', rotation=30)

        bars2 = ax2.bar(comp['Model'], comp['AUC-ROC'], color=colors_bar[:len(comp)])
        ax2.set_title('AUC-ROC Comparison')
        ax2.set_ylabel('AUC-ROC')
        ax2.set_ylim(0, 1.1)
        for bar, auc_val in zip(bars2, comp['AUC-ROC']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{auc_val:.3f}', ha='center', fontsize=9)
        ax2.tick_params(axis='x', rotation=30)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ANN Training History
        if 'history' in artifacts:
            st.subheader("ANN Training History")
            hist = artifacts['history']

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(hist['accuracy'], label='Train')
            ax1.plot(hist['val_accuracy'], label='Validation')
            ax1.set_title('ANN Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(hist['loss'], label='Train')
            ax2.plot(hist['val_loss'], label='Validation')
            ax2.set_title('ANN Loss')
            ax2.set_xlabel('Epoch')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Feature Importance
        if 'rf_model' in artifacts:
            st.subheader("Feature Importance (Random Forest)")
            feature_names = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
                           'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY',
                           'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING',
                           'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']
            importances = artifacts['rf_model'].feature_importances_

            fi = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            fi = fi.sort_values('Importance', ascending=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(fi['Feature'], fi['Importance'], color='steelblue')
            ax.set_title('Feature Importance')
            ax.set_xlabel('Importance Score')
            st.pyplot(fig)
            plt.close()


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("*Lung Cancer Prediction using ML & DL | PBEL NASCOM Internship | N. Mohammed Sohaib (74)*")
