import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"e:\final_sales_report.txt")

plt.figure(figsize=(10, 6))

sns.barplot(data=df, x='City', y='Total_Sales', estimator=sum, errorbar=None, palette='viridis')

plt.title('Total Sales per City', fontsize=16)
plt.xlabel('City', fontsize=12)
plt.ylabel('Sales (EGP)', fontsize=12)
plt.grid(axis='y', ls='--', alpha=0.7)

plt.show()