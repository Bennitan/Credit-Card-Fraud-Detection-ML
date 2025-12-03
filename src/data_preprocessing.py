import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv('C:\\Users\\benni\\OneDrive\\Desktop\\Fraud_Detection_Project\\data\\creditcard.csv')

# Check first 5 rows
print(data.head())

# Check dataset info
print(data.info())

# Check for missing values
print(data.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# Count of fraud vs non-fraud
sns.countplot(x='Class', data=data)
plt.title('Distribution of Fraud vs Non-Fraud')
plt.show()


# Calculate and print the percentage of fraud cases (Class = 1)
fraud_count = data['Class'].value_counts()[1]
non_fraud_count = data['Class'].value_counts()[0]
total_count = len(data)

print("\n--- Class Imbalance Metrics ---")
print(f"Total Transactions: {total_count}")
print(f"Non-Fraud Transactions (0): {non_fraud_count}")
print(f"Fraud Transactions (1): {fraud_count}")

# Use floating point division for accurate percentage
fraud_percentage = (fraud_count / total_count) * 100
print(f"Fraud Percentage: {fraud_percentage:.4f}%")
print("This extreme imbalance (less than 0.2%) is the core challenge of this project and must be addressed.") 


from sklearn.model_selection import train_test_split

print("\n--- Splitting Data into Training and Testing Sets ---")

# 1. Define X (Features) and y (Target)
X = data.drop('Class', axis=1) # Drop the 'Class' column from features
y = data['Class']              # The 'Class' column is the target

# 2. Split the data
# test_size=0.2 means 20% of the data goes to the test set.
# random_state=42 ensures the split is the same every time we run the code.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"X_train Shape (80% of data): {X_train.shape}")
print(f"X_test Shape (20% of data): {X_test.shape}")
print(f"y_train Shape: {y_train.shape}")
print(f"y_test Shape: {y_test.shape}") 

# data_preprocessing.py

# ... (Previous code ends here after the data splitting print statements) ...

from imblearn.over_sampling import SMOTE
from collections import Counter

print("\n--- Applying SMOTE to Balance Training Data ---")

# 1. Check the imbalance BEFORE SMOTE
print("Original Training Set shape %s" % Counter(y_train))

# 2. Initialize SMOTE
# random_state ensures the synthetic samples are the same every time we run the code.
sm = SMOTE(random_state=42)

# 3. Apply SMOTE to the training features (X_train) and target (y_train)
# The output is the new, balanced training set.
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# 4. Check the balance AFTER SMOTE
print("Resampled Training Set shape %s" % Counter(y_train_sm)) 

# Now, X_train_sm and y_train_sm are the balanced datasets we will use for training!