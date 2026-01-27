# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
sns.set_theme(style="whitegrid")

# 2. Load Data
file_path = r"c:\Users\Hp\Desktop\data_science2\titanic.txt"
df = pd.read_csv(file_path)

age_mean = df['age'].mean()
df['age'] = df['age'].fillna(age_mean)

df = df.dropna(subset=['embark_town'])

# ==========================================================
# 3. Feature Engineering
# ==========================================================
df['who_code'] = df['who'].map({'man': 0, 'woman': 1, 'child': 2})

df['FamilySize'] = df['sibsp'] + df['parch'] + 1

df.rename(columns={'alone': 'IsAlone'})
df['IsAlone'] = 0
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
print(df.head())

cols_to_drop = ['survived', 'alive', 'embark_town', 'adult_male', 'class', 'deck',
                'who', 'sex', 'sibsp', 'parch', 'embarked', 'alone']

X = df.drop(cols_to_drop, axis=1, errors='ignore')
y = df['survived']

print("Columns which Mode will Study:")
print(X.columns)
print("\nForm of Data:", X.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nX_train shape (for Study):", X_train.shape)
print("X_test shape (for Exam):", X_test.shape)
print("y_train shape (Answer of Study):", y_train.shape)
print("y_test shape (Answer of Exam):", y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of Model: {accuracy * 100:.2f}%")

final_df = pd.DataFrame({
  'PassengerId': X_test['Unnamed: 0'],
  'survived': y_pred
})
final_df.to_csv(r"c:\Users\Hp\Desktop\data_science2\titanic_submission.csv.txt", index=False)