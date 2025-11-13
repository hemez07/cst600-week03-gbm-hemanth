# 🩺 Disease Diagnosis with Gradient Boosting  
**Course:** CST600 — Week 3 Assignment  
**Student:** Hemanth
**Date:** November 2025  

---

## 🎯 Objective
The goal of this assignment is to develop a **machine learning model** using **Gradient Boosting (GBM)** to predict whether a breast tumor is **benign (B)** or **malignant (M)**.  
The project demonstrates end-to-end workflow: **EDA → Preprocessing → Modeling → Evaluation → Interpretation** using Python and scikit-learn.

---

## 📦 Dataset
**Source:**  
[Kaggle — Breast Cancer Wisconsin (Diagnostic) Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)  
Originally published by the **UCI Machine Learning Repository**.

**Description:**  
The dataset contains **569 samples** and **30 numeric features** describing cell nuclei characteristics derived from breast mass images.

| Column Type | Examples |
|--------------|-----------|
| Measurement | `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean` |
| Shape Irregularity | `concavity_mean`, `concave points_mean` |
| Texture | `smoothness_mean`, `compactness_mean` |
| Target | `diagnosis` — Malignant (M) or Benign (B) |

---

## 🔍 Exploratory Data Analysis (EDA)
EDA was performed to explore structure, class balance, and feature relationships.

**Key findings:**
- Dataset shape: **569 rows × 31 columns**  
- Target distribution: **37% malignant**, **63% benign**
- No missing values found
- Features such as `radius_mean`, `area_mean`, and `perimeter_mean` are strongly correlated

**Visuals saved to `/figures/`:**
- `corr_heatmap.png` — correlation between numeric variables  
- `hist_radius_mean.png` — distribution of tumor radius  
- `countplot_diagnosis.png` — class balance visualization  

---

## ⚙️ Data Preprocessing
All preprocessing steps were handled in Python scripts located in `/src/`.

**Steps performed:**
1. Dropped non-informative columns: `id`, `Unnamed: 32`  
2. Encoded target variable: `diagnosis` → M = 1, B = 0  
3. Train/test split (80/20) with stratification for class balance  
4. No scaling required (GBM is tree-based)  
5. Saved processed dataset to:  
   `data/processed/breast_cancer_clean.csv`

---

## 🤖 Model — Gradient Boosting Classifier
**Library:** `sklearn.ensemble.GradientBoostingClassifier`

**Chosen hyperparameters:**

| Parameter | Value | Justification |
|------------|--------|----------------|
| `n_estimators` | 200 | Ensures convergence |
| `learning_rate` | 0.1 | Balances bias-variance trade-off |
| `max_depth` | 3 | Prevents overfitting |
| `subsample` | 0.8 | Adds randomness for robustness |
| `random_state` | 42 | Reproducibility |
| `n_iter_no_change` | 10 | Enables early stopping |

---

## 🧪 Model Evaluation

### 🔹 5-Fold Cross-Validation
| Metric | Mean | Std |
|---------|------|------|
| Accuracy | **0.9670** | 0.0139 |
| ROC-AUC | **0.9915** | 0.0050 |

---

### 🔹 Test Set Performance
| Metric | Score |
|---------|--------|
| Accuracy | **0.9649** |
| Precision | **1.0000** |
| Recall | **0.9048** |
| F1-Score | **0.9500** |
| ROC-AUC | **0.9970** |

**Confusion Matrix:**
|                | Predicted Benign | Predicted Malignant |
|----------------|------------------|----------------------|
| **Actual Benign** | 72 | 0 |
| **Actual Malignant** | 4 | 38 |

**Interpretation:**
- No false positives (perfect precision)
- 4 false negatives — recall slightly below 1.0
- Overall accuracy ≈ **96%**  
- Model demonstrates excellent generalization and discrimination (AUC ≈ 0.997)

---

## 💡 Feature Importance
Top predictors influencing diagnosis:
1. `worst concave points`  
2. `worst perimeter`  
3. `worst radius`  
4. `mean concave points`  
5. `mean radius`

**Interpretation (for healthcare stakeholders):**  
Tumor **shape irregularity and size** are key indicators of malignancy.  
These results align with clinical observations that irregular, large tumors tend to be malignant.

Visual saved as: `figures/top10_features.png`

---

## ⚖️ Responsible Modeling
- **Sensitive attributes:** None included (no patient demographics)  
- **Data leakage:** Prevented by clean feature-target separation  
- **Bias mitigation:** Used stratified split for balanced training/testing  
- **Limitations:**  
  - Dataset from a single medical source (may not generalize globally)  
  - Small sample size for rare malignant cases  

---

## 📁 Project Structure

cst600-week03-gbm-hemanth/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│ ├── raw/data.csv
│ └── processed/breast_cancer_clean.csv
│
├── src/
│ ├── main.py
│ ├── preprocessing.py
│ └── model_gbm.py
│
├── outputs/
│ ├── gbm_model.joblib
│ ├── metrics.json
│ └── feature_importances.csv
│
├── figures/
│ ├── corr_heatmap.png
│ ├── roc_curve.png
│ ├── confusion_matrix.png
│ └── top10_features.png
│
└── ppt/
└── week03_gbm_presentation.pptx


---

## 🧰 Environment Setup & Running

```bash
# 1. Clone the repository
git clone https://github.com/<hemez07>/cst600-week03-gbm-hemanth.git
cd cst600-week03-gbm-hemanth

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run main script
python src/main.py


Outputs are automatically saved to /outputs/ and /figures/.