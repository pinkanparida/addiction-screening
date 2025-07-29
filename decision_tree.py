# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the cleaned dataset
df = pd.read_csv("C:\\Users\\apkon\\OneDrive\\Desktop\\Behavioural Screening\\screening_data.csv") # use your actual file name
df.fillna(0, inplace=True) 

# Step 3: Calculate total score (if not already present)
df['Total_Score'] = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5',
                        'Q6', 'Q7', 'Q8', 'Q9', 'Q10',
                        'Q11', 'Q12', 'Q13', 'Q14', 'Q15']].sum(axis=1)

# Step 4: Create a label/category based on score
def classify(score):
    if score >= 12:
        return 'Addicted'
    elif score >= 7:
        return 'Moderate'
    else:
        return 'Non-Addicted'

df['Category'] = df['Total_Score'].apply(classify)

# Step 5: Prepare features (X) and target (y)
X = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5',
        'Q6', 'Q7', 'Q8', 'Q9', 'Q10',
        'Q11', 'Q12', 'Q13', 'Q14', 'Q15']]
y = df['Category']

# Step 6: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Initialize and train the Decision Tree model
dtree = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
dtree.fit(X_train, y_train)

# Step 8: Predict on test data
y_pred = dtree.predict(X_test)

# Step 9: Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=dtree.classes_, yticklabels=dtree.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tree Confusion Matrix")
plt.show()

from sklearn.tree import export_text
print(export_text(dtree, feature_names=list(X.columns)))

from sklearn import tree
plt.figure(figsize=(20, 10))
tree.plot_tree(dtree, feature_names=X.columns, class_names=dtree.classes_, filled=True)
plt.show()
