# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# 2. Load Data
file_path = r"c:\Users\Hp\Desktop\data_science2\titanic.txt"
df = pd.read_csv(file_path)

# 3. Data Cleaning
df = df.drop('deck', axis=1)

age_mean = df['age'].mean()
df['age'] = df['age'].fillna(age_mean)

df = df.dropna(subset=['embark_town'])

print("Data Cleaned! Remaining Nulls:", df.isnull().sum().sum())

# 4. Data Visualization
plt.figure(figsize=(6, 4))
sns.countplot(x='survived', data=df, palette='pastel')
plt.title('Distribution ofSurvival (0=Died, 1=Survived)')
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(x='sex', y='survived', data=df, palette='coolwarm')
plt.title("Survival Probability")
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(x='pclass', y='survived', data=df, palette='viridis')
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Probability')
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['age'], bins=30, kde=True, color='purple')
plt.title('Age Distribution of Passengers')
plt.show()