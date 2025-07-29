# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load your dataset (update the file name if needed)
df = pd.read_csv("C:\\Users\\apkon\\OneDrive\\Desktop\\Behavioural Screening\\screening_data.csv") # use your actual file name
df.fillna(0, inplace=True)
# Step 3: Calculate total score
df['Total_Score'] = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5',
                        'Q6', 'Q7', 'Q8', 'Q9', 'Q10',
                        'Q11', 'Q12', 'Q13', 'Q14', 'Q15']].sum(axis=1)

# Step 4: Classify into categories
def classify(score):
    if score >= 12:
        return 'Addicted'
    elif score >= 7:
        return 'Moderate'
    else:
        return 'Non-Addicted'

df['Category'] = df['Total_Score'].apply(classify)

# Step 5: Prepare features (X) and labels (y)
X = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5',
        'Q6', 'Q7', 'Q8', 'Q9', 'Q10',
        'Q11', 'Q12', 'Q13', 'Q14', 'Q15']]
y = df['Category']

# Step 6: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Step 8: Make predictions on test data
y_pred = knn.predict(X_test)

# Step 9: Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Optional - Visualize confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=knn.classes_, yticklabels=knn.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN Confusion Matrix")
plt.show()
