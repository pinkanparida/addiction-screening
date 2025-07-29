# Step 1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the dataset
df = pd.read_csv("C:\\Users\\apkon\\OneDrive\\Desktop\\Behavioural Screening\\screening_data.csv") # use your actual file name
df.fillna(0, inplace=True)  # use your actual file

# Step 3: Calculate Total Score (if not already present)
df['Total_Score'] = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5',
                        'Q6', 'Q7', 'Q8', 'Q9', 'Q10',
                        'Q11', 'Q12', 'Q13', 'Q14', 'Q15']].sum(axis=1)

# Step 4: Create Category based on score
def classify(score):
    if score >= 12:
        return 'Addicted'
    elif score >= 7:
        return 'Moderate'
    else:
        return 'Non-Addicted'

df['Category'] = df['Total_Score'].apply(classify)

# Step 5: Prepare features and target
X = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5',
        'Q6', 'Q7', 'Q8', 'Q9', 'Q10',
        'Q11', 'Q12', 'Q13', 'Q14', 'Q15']]
y = df['Category']

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Initialize and train SVM model
svm_model = SVC(kernel='linear')  # You can try 'rbf', 'poly', etc.
svm_model.fit(X_train, y_train)

# Step 8: Predict on test data
y_pred = svm_model.predict(X_test)

# Step 9: Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=svm_model.classes_, yticklabels=svm_model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")
plt.show()

df_test = X_test.copy()
df_test['Actual'] = y_test
df_test['Predicted'] = y_pred
df_test.to_excel("C:\\Users\\apkon\\OneDrive\\Desktop\\Behavioural Screening\\svm_predictions.xlsx", index=False)
