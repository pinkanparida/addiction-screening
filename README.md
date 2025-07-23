# 🧠 Behavioral Screening System – Addiction Detection via Survey

A Machine Learning-based system to detect potential behavioral addiction using survey data, classification models, and visual analytics.

---

## 📌 Project Overview

This project focuses on building a behavioral addiction screening system using survey responses. It aims to classify individuals into **Addicted** or **Non-Addicted** categories using Machine Learning algorithms based on their survey scores and response patterns.

---

## 🔧 Tools & Technologies

- **Language**: Python
- **IDE**: PyCharm Community Edition 2024.2
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib  
- **Other Tools**: Excel (for data cleaning)  
- **ML Models**: SVM, KNN, Decision Tree  
- **Evaluation**: Confusion Matrix, Accuracy, Precision, Recall  
- **Visualizations**: Correlation matrix, Age vs Addiction graphs  

---

## 🗂️ Dataset

- **Source**: Self-designed behavioral survey
- **Format**: CSV
- **Features**:
  - Demographics (Age, Gender)
  - Behavioral questions with Likert-scale responses
  - Final score
  - Labeled output: Addicted / Non-Addicted

---

## 🎯 Objectives

- Preprocess and clean survey data using Excel & Pandas.
- Generate a scoring system to classify participants.
- Apply and compare multiple ML algorithms (SVM, KNN, Decision Tree).
- Visualize correlation and distribution across groups (age-wise trends).
- Evaluate performance with confusion matrices and classification reports.

---

## 📊 Key Visualizations

- **Correlation Matrix** – Shows relationship between different survey features.
- **Age-wise Line Graph** – Addiction trends across age groups.
- **Confusion Matrix** – Model performance visualization.

_Screenshots or plots can be inserted here if available._

---

## 🚀 Project Workflow

1. **Data Collection**: Survey responses gathered and stored in CSV.
2. **Data Preprocessing**:
   - Cleaned using Excel and Pandas
   - Label encoding for categorical responses
3. **Scoring System**:
   - Each response mapped to numerical score
   - Final score used for classification
4. **Model Building**:
   - Used SVM, KNN, Decision Tree
   - Trained and tested on split dataset
5. **Evaluation**:
   - Compared model accuracy and confusion matrix
   - Used classification report for precision, recall, F1-score
6. **Visualization**:
   - Age-based trend analysis
   - Correlation heatmap using Seaborn/Matplotlib

---

## ✅ Results

| Model          | Accuracy |
|----------------|----------|
| SVM            | 85%      |
| KNN            | 83%      |
| Decision Tree  | 87%      |

> Decision Tree performed the best in terms of accuracy and interpretability.

---

## 📁 Project Structure

```bash
addiction-screening/
│
├── data/
│   └── survey_data.csv
│
├── visuals/
│   └── correlation_matrix.png
│   └── age_vs_addiction.png
│
├── README.md
└── requirements.txt
