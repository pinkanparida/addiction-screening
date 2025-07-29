import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv("C:\\Users\\apkon\\OneDrive\\Desktop\\Behavioural Screening\\screening_data.csv")
df.fillna(0, inplace=True)
df['Total_Score'] = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5',
                        'Q6', 'Q7', 'Q8', 'Q9', 'Q10',
                        'Q11', 'Q12', 'Q13', 'Q14', 'Q15']].sum(axis=1)
def classify(score):
    if score >= 12:
        return 'Addicted'
    elif score >= 7:
        return 'Moderate'
    else:
        return 'Non-Addicted'

df['Category'] = df['Total_Score'].apply(classify)
sns.countplot(x='Category', data=df)
plt.title("Distribution of Categories")
plt.xlabel("Category")
plt.ylabel("Number of Students")
plt.show()
X = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5',
        'Q6', 'Q7', 'Q8', 'Q9', 'Q10',
        'Q11', 'Q12', 'Q13', 'Q14', 'Q15']]
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
df.to_excel("C:\\Users\\apkon\\OneDrive\\Desktop\\Behavioural Screening\\screening_with_category.xlsx", index=False)


