# 🧬 Liver Cirrhosis Stage Detection

## 📘 Overview

This project focuses on predicting the **stage of liver cirrhosis** in patients using machine learning. The entire pipeline is implemented in the Jupyter Notebook **`liver_project.ipynb`**, which covers data preprocessing, feature encoding, scaling, and classification using the **XGBoost Classifier**. The model predicts one of three liver cirrhosis stages based on clinical data.

---

## 🎯 Objective

To build a reliable multi-class classification system that can predict whether a patient is in Stage 1, 2, or 3 of liver cirrhosis based on clinical and diagnostic information.

---

## 📂 Dataset

The dataset used is `liver_cirrhosis.csv`, which contains clinical attributes of liver disease patients including their diagnostic test results and treatment indicators.

> Make sure the CSV file is in the same directory as the notebook for correct execution.

---

## ⚙️ Technical Workflow

### 🔄 Preprocessing

- Categorical variables (`Status`, `Drug`, `Sex`, `Ascites`, `Hepatomegaly`, `Spiders`, `Edema`) are encoded using `LabelEncoder`.
- Numerical features are standardized using `StandardScaler`.
- Labels are adjusted from original values [1, 2, 3] to [0, 1, 2] for model training.

### 🧠 Modeling

- Model used: **XGBoost Classifier** (`XGBClassifier`)
- Objective: `multi:softmax` for multiclass classification
- Evaluation Metric: `mlogloss`

```python
xgb = XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', random_state=42)
