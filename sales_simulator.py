import numpy as np

# 1. Data Setup
np.random.seed(42)

# Sales Matrix for Company
sales_matrix = np.random.randint(10000, 100000, size=(5, 6))
print(f"Original Sales Matrix:\n {sales_matrix}")

# 2. Aggregation
sales_per_branch = np.sum(sales_matrix,axis=1)
sales_per_month = np.sum(sales_matrix,axis=0)

print(f"\nTotal sales per Branch: {sales_per_branch}")
print(f"Total sales in March (Month 3): {sales_per_month[2]}")

# 3. Performance Analysis
best_branch_sales = np.max(sales_per_branch)
best_branch_index = np.argmax(sales_per_branch)

print(f"\nBest Performing Branch: Branch {best_branch_index + 1}")
print(f"Highest Sales Value: {best_branch_sales}")

best_month_index = np.argmax(sales_per_month)
print(f"\nBest Performing Month: Month {best_month_index + 1}")

# 4. Tax & Discounts
TAX_RATE = 0.9
sales_after_tax = sales_matrix * TAX_RATE
print(f"\nSales Matrix after 10% Tax:\n {sales_after_tax}\n")

# 5. Critical Issues
bad_branches_indices, bad_months_indices = np.where(sales_matrix < 20000)

for branch_idx, month_idx in zip(bad_branches_indices, bad_months_indices):
  value = sales_matrix[branch_idx, month_idx]
  print(f"⚠ Alert: Branch {branch_idx + 1} failed in Month {month_idx + 1} with sales: {value}")

print(f"Total Incidents: {len(bad_branches_indices)}")
