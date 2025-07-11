import pandas as pd
import os

# Check file existence and sizes
print("Checking files...")
files = ['train_data.csv', 'test_data.csv', 'train_labels.csv', 'sample_submission.csv']
for file in files:
    if os.path.exists(file):
        size_gb = os.path.getsize(file) / (1024**3)
        print(f"{file}: {size_gb:.2f} GB")
    else:
        print(f"{file}: Not found")

# Quick peek at data structure
try:
    print("\n--- Sample Submission ---")
    sample_sub = pd.read_csv('sample_submission.csv')
    print(f"Shape: {sample_sub.shape}")
    print(f"Columns: {list(sample_sub.columns)}")
    print(sample_sub.head())

    print("\n--- Train Labels (first 10 rows) ---")
    labels = pd.read_csv('train_labels.csv', nrows=10)
    print(f"Shape: {labels.shape}")
    print(f"Columns: {list(labels.columns)}")
    print(labels.head())

    print("\n--- Train Data (first 5 rows, first 10 columns) ---")
    train = pd.read_csv('train_data.csv', nrows=5)
    print(f"Shape: {train.shape}")
    print(f"Total columns: {len(train.columns)}")
    print(f"First 10 columns: {list(train.columns[:10])}")
    print(train.iloc[:, :10].head())

    # Check if this is American Express data format
    if 'customer_ID' in train.columns and 'S_2' in train.columns:
        print("\n--- This appears to be American Express Default Prediction data ---")
        print("Time series format detected")

        # Count unique customers
        unique_customers = train['customer_ID'].nunique()
        print(f"Unique customers in sample: {unique_customers}")

        # Check statement dates
        if 'S_2' in train.columns:
            print(f"Statement dates range: {train['S_2'].min()} to {train['S_2'].max()}")

except Exception as e:
    print(f"Error: {e}")

print("\nData check completed.")
