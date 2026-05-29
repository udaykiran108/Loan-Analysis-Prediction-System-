import pandas as pd

df = pd.read_csv("train_u.csv")
print(df.head())

print("Before cleaning:")
print(df.isnull().sum())

# Cleaning
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())

df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

# Fix Dependents
df['Dependents'] = df['Dependents'].replace('3+', 3)
df['Dependents'] = df['Dependents'].astype(int)

print("After cleaning:")
print(df.isnull().sum())

#EDA
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
sns.countplot(x='Loan_Status', data=df)
plt.title("Loan Status")

plt.subplot(2,2,2)
sns.countplot(x='Gender', hue='Loan_Status', data=df)
plt.title("Gender vs Loan")

plt.subplot(2,2,3)
sns.countplot(x='Credit_History', hue='Loan_Status', data=df)
plt.title("Credit History")

plt.subplot(2,2,4)
sns.histplot(df['ApplicantIncome'], kde=True)
plt.title("Income")

plt.tight_layout()
plt.show()

# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)  # features
y = df['Loan_Status']  # target
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(confusion_matrix(y_test, y_pred))