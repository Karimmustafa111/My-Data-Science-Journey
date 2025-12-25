import numpy as np
import pandas as pd

# =============================================
# 1. Data Loading
# =============================================
sales_df = pd.read_csv("e:\supermarket_sales.txt")

print("--- Data Overview ---")
print(sales_df.head())
print(f"Data Shape: {sales_df.shape}")

# =============================================
# 2. Feature Engineering
# =============================================
sales_df['Total_Sales'] = sales_df['Price'] * sales_df['Quantity']

print("\n--- Added Total_Sales Column ---")
print(sales_df[['Price', 'Quantity', 'Total_Sales']])

print("\n--- Description of Tabel ---")
print(sales_df.describe())

# =============================================
# 3. City Analysis
# =============================================
city_sales_summary = sales_df.groupby("City")['Total_Sales'].sum()

top_cities = city_sales_summary.sort_values(ascending=False)

print("\n--- Best Performing Cities ---")
print(top_cities)

# =============================================
# 4. Payment Analysis
# =============================================
payment_stats = sales_df['Payment_Method'].value_counts()

print("\n--- Payment Methods Usage ---")
print(payment_stats)

# =============================================
# 5. Advanced Filtering
# =============================================
high_value_cairo_txns = sales_df[
  (sales_df['City'] == 'Cairo') & 
  (sales_df['Total_Sales'] > 100000)
]

print(f"\n--- High Value Transactions in Cairo: {len(high_value_cairo_txns)} ---")

# =============================================
# 6. pivot Analysis
# =============================================
city_product_report = sales_df.groupby(["City", "Product"])['Quantity'].sum()

print("\n--- Product Sales by City ---")
print(city_product_report.head(10))

# =============================================
# 7. Categorization
# =============================================
average_sales = sales_df['Total_Sales'].mean()

sales_df['Sales_Category'] = np.where(
  sales_df['Total_Sales'] > average_sales,
  'High_Value',
  'Standerd'
)

print("\n--- Sales Categories Distribution ---")
print(sales_df['Sales_Category'].value_counts())

# =============================================
# 8. Export
# =============================================
sales_df.to_csv(r"e:\final_sales_report.txt",index=False)
print("\n✔ Report saved successfully to 'final_sales_report.csv'")