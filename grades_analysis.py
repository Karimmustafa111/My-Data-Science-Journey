import numpy as np

np.random.seed(42)
grades = np.random.randint(10, 100, 30)

average = np.mean(grades)
max_grade = np.max(grades)
min_grade = np.min(grades)

print(f"Average of grades: {average}")
print(f"Max grade: {max_grade}")
print(f"Min grade: {min_grade}")

passing_grades = grades[grades >= 50]
print(f"Successful grades: {passing_grades}")
print(f"Number of successed students: {len(passing_grades)}")