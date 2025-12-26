import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')

# =============================================
# 1. Data Preparation
# =============================================
browsers = ['Chrome', 'Firefox', 'Safari', 'Edge']
carts = [0, 1]
age_groups = ['Teens', 'Seniors', 'Adults']

np.random.seed(42)
data = {
  'Visitor_ID': np.arange(1, 1001),
  'Browser': np.random.choice(browsers, 1000),
  'Time_Spent_Seconds': np.abs(np.random.normal(loc=180, scale=60, size=1000)),
  'Add_to_Cart': np.random.choice(carts, 1000),
  'Age_Group': np.random.choice(age_groups, 1000)
}

df = pd.DataFrame(data)

# ==============================================
# 2. Feature Engineering
# ==============================================
df['Conversion_Score'] = (df['Time_Spent_Seconds'] * 0.5) + (df['Add_to_Cart'] * 50)

df['Conversion_Score'] = df['Conversion_Score'].round(1)
df['Time_Spent_Seconds'] = df['Time_Spent_Seconds'].round(1)

print("--- Data Sample ---")
print(df.head())
print("\n" + "="*40 + "\n")

# ==============================================
# 3. Visualization
# ==============================================

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Browser', y='Time_Spent_Seconds', palette='viridis', linewidth=1.5)
plt.title('📊 Analysis: Time Spent per Browser Type', fontsize=14, fontweight='bold', color='#333333')
plt.xlabel('Browser Type', fontsize=12)
plt.ylabel('Time Spent (Seconds)', fontsize=12)

plt.savefig('browser_time_analysis.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='Age_Group', palette='pastel', order=['Teens', 'Adults', 'Seniors'])

plt.title('👥 User Distribution by Age Group', fontsize=14, fontweight='bold')
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Count of Users', fontsize=12)

for p in ax.patches:
  ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
              ha='center', va='bottom', fontsize=11, color='k', xytext=(0, 5),
              textcoords='offset points')
sns.despine(left=True)

plt.savefig('age_distribution.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(9, 7))
correlation = df.corr(numeric_only=True)

sns.heatmap(data=correlation, annot=True, cmap='RdBu_r', fmt='.2f',
            linewidths=0.5, vmin=1, vmax=1, center=0, cbar_kws={"shrink": .8})
plt.title('🔥 Correlation Matrix (Relationships)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=45)

plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')