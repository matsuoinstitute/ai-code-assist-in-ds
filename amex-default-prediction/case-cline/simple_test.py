import pandas as pd
import numpy as np

print("Starting simple data test...")

try:
    # Sample submission check
    sample_sub = pd.read_csv('sample_submission.csv')
    print(f"Sample submission shape: {sample_sub.shape}")
    print(f"Sample submission columns: {list(sample_sub.columns)}")
    print("Sample submission head:")
    print(sample_sub.head())

    # Train labels check
    train_labels = pd.read_csv('train_labels.csv')
    print(f"\nTrain labels shape: {train_labels.shape}")
    print(f"Train labels columns: {list(train_labels.columns)}")
    print("Train labels head:")
    print(train_labels.head())

    # Target distribution
    print(f"\nTarget distribution:")
    print(train_labels['target'].value_counts())
    print(f"Target rate: {train_labels['target'].mean():.4f}")

    # Small sample of train data
    print(f"\nLoading small sample of train data...")
    train_sample = pd.read_csv('train_data.csv', nrows=100)
    print(f"Train sample shape: {train_sample.shape}")
    print(f"Train sample columns (first 10): {list(train_sample.columns[:10])}")

    # Check for customer_ID and S_2 columns
    if 'customer_ID' in train_sample.columns:
        print(f"Unique customers in sample: {train_sample['customer_ID'].nunique()}")

    if 'S_2' in train_sample.columns:
        print(f"S_2 column type: {train_sample['S_2'].dtype}")
        print(f"S_2 sample values: {train_sample['S_2'].head().tolist()}")

    print("\nData test completed successfully!")

except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
