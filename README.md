# Lung Cancer Prediction using Machine Learning & Deep Learning

**PBEL NASCOM Internship – AI/ML Project**  
**By: N. Mohammed Sohaib (74)**

## Overview
Predict lung cancer risk from patient health survey data using 5 Machine Learning models + 1 Deep Learning (ANN) model. Includes an interactive Streamlit web app for real-time predictions.

## Project Structure
| File | Description |
|------|-------------|
| `74_N_Mohammed_Sohaib_Lung_Cancer.ipynb` | Complete Jupyter Notebook (EDA → Train → Evaluate) |
| `lung_cancer_data.csv` | Dataset — 1,000 patient records, 15 features |
| `streamlit_app.py` | Interactive Streamlit web application |
| `random_forest_model.pkl` | Trained Random Forest model |
| `ann_model.h5` | Trained ANN (Deep Learning) model |
| `Lung_Cancer_Prediction.pptx` | Project presentation (14 slides) |
| `train_model.py` | Standalone training script |
| `download_data.py` | Dataset generation script |
| `generate_pptx.py` | Presentation generator |

## Models Compared
| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Logistic Regression | 74.0% | 0.827 |
| Random Forest | 74.0% | 0.796 |
| SVM | 73.5% | 0.794 |
| ANN (Deep Learning) | 72.5% | 0.773 |
| KNN | 67.0% | 0.702 |
| Decision Tree | 64.5% | 0.643 |

## Tech Stack
- Python, TensorFlow/Keras, Scikit-Learn
- Pandas, NumPy, Matplotlib, Seaborn
- Streamlit (Web GUI)

## How to Run

### Notebook
Open `74_N_Mohammed_Sohaib_Lung_Cancer.ipynb` in Jupyter and run all cells.

### Streamlit App
```bash
pip install streamlit tensorflow pandas numpy matplotlib seaborn scikit-learn
streamlit run streamlit_app.py
```

## Key Features
- 15 patient attributes: Age, Gender, Smoking, Coughing, Chest Pain, etc.
- Binary classification: Lung Cancer (YES / NO)
- Feature importance analysis using Random Forest
- Comprehensive EDA with correlation heatmaps
- Real-time patient risk prediction via Streamlit GUI
