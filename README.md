# 🧠 Behavioural Screening Using Machine Learning

This project focuses on identifying behavioural patterns among 300 male students based on their responses to 15 sensitive questions regarding their attitudes and interest toward pornography. The goal is to classify individuals into categories such as **Addicted**, **Moderate**, or **Non-Addicted** using Machine Learning models.

---

## 📂 Project Overview

- 🎯 **Goal:** Early identification of potential pornography addiction through behavioural indicators.
- 📊 **Dataset:** 300 male students’ responses to 15 behavior-related questions.
- ✍️ **Type of Data:** Categorical (Yes/No, Agree/Disagree/Strongly Agree) converted into numerical form.
- 🧮 **Scoring Method:** Total score out of 15, used to classify behavioural categories.
- 🤖 **ML Algorithms Used:**
  - K-Nearest Neighbors (KNN)
  - Decision Tree Classifier
  - Support Vector Machine (SVM)

---

## 📈 Scoring System

Each participant's 15 responses were numerically encoded and summed:

- **Total Score ≥ 12** → Addicted  
- **Total Score between 7–11** → Moderate  
- **Total Score < 7** → Non-Addicted  

This score was used as the target label (`Category`) for model training.

---

## 🧠 Machine Learning Models

### 🔹 K-Nearest Neighbors (KNN)
- Used to predict the addiction category based on proximity to similar students.
- Accuracy: *Reported after evaluation.*

### 🔹 Decision Tree Classifier
- Easy to interpret model for splitting students into categories.
- Includes confusion matrix and classification report.

### 🔹 Support Vector Machine (SVM)
- Used with linear kernel for high-dimensional separation.
- Compared with other models for accuracy and performance.

---

## ⚙️ Technologies & Tools

- **Language:** Python  
- **IDE:** PyCharm  
- **Libraries:**
  - `pandas`, `numpy`
  - `scikit-learn`
  - `matplotlib`, `seaborn`
  - `openpyxl` for Excel I/O

---

## 🗂️ Folder Structure

```plaintext
📁 behavioural-screening/
├── behavioural_scoring.py         # Preprocessing and scoring
├── knn_classifier.py             # KNN model code
├── decision_tree_classifier.py   # Decision Tree model code
├── svm_classifier.py             # SVM model code
├── screening_data.xlsx           # Raw/cleaned dataset
├── screening_with_category.xlsx  # Final scored data with labels
└── README.md                     # Project documentation
