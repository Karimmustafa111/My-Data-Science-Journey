# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
sns.set_theme(style="whitegrid")

# 2. Load Data
file_path = r"c:\Users\Hp\Desktop\data_science2\titanic.txt"
df = pd.read_csv(file_path)

# 3. Data Cleaning
df = df.drop('deck', axis=1)

age_mean = df['age'].mean()
df['age'] = df['age'].fillna(age_mean)

df = df.dropna(subset=['embark_town'])

# print("Data Cleaned! Remaining Nulls:", df.isnull().sum().sum())

# # 4. Data Visualization
# plt.figure(figsize=(6, 4))
# sns.countplot(x='survived', data=df, palette='pastel')
# plt.title('Distribution ofSurvival (0=Died, 1=Survived)')
# plt.show()

# plt.figure(figsize=(6, 4))
# sns.barplot(x='sex', y='survived', data=df, palette='coolwarm')
# plt.title("Survival Probability")
# plt.show()

# plt.figure(figsize=(6, 4))
# sns.barplot(x='pclass', y='survived', data=df, palette='viridis')
# plt.title('Survival Rate by Passenger Class')
# plt.ylabel('Survival Probability')
# plt.show()

# plt.figure(figsize=(8, 5))
# sns.histplot(df['age'], bins=30, kde=True, color='purple')
# plt.title('Age Distribution of Passengers')
# plt.show()

# ==========================================================
# Part Two: Data Preparation (Machine Learning)
# ==========================================================
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print("Data after Replacing:")
print(df.head())

# ==========================================================
# Data Division
# ==========================================================
X = df.drop(['survived', 'alive', 'who', 'embark_town', 'adult_male', 'class'], axis=1)

y = df['survived']

# print("\nForm of Question: ", X.shape)
# print("Form of answer: ", y.shape)

# ==========================================================
# 4. Data Division (Train & Test) [Split]
# ==========================================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nX_train shape (for Study):", X_train.shape)
print("X_test shape (for Exam):", X_test.shape)
print("y_train shape (Answer of Study):", y_train.shape)
print("y_test shape (Answer of Exam):", y_test.shape)

# ==========================================================
# 5. Build the Model and (Training)
# ==========================================================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=1000)

print("⌛")
model.fit(X_train, y_train)
print("✅ Trained Successfully")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of Model: {accuracy * 100:.2f}%")

# =========================================================
# 5. Feature Importance
# =========================================================

feature_names = X.columns

coefficients = model.coef_[0]

importance_df = pd.DataFrame({
  'Feature': feature_names,
  'Importance': coefficients
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nThe Most Important Efficint Features in Survival (with Number):")
print(importance_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance: What mattered most?')
plt.xlabel('Impact on Survival (Positive = Good, Negative = Bad)')
plt.show()