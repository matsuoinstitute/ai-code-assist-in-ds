import pandas as pd
import numpy as np
import sys

# 出力をファイルにリダイレクト
with open('analysis_results.txt', 'w') as f:
    sys.stdout = f

    print("=== American Express Default Prediction Data Analysis ===")

    try:
        # Sample submission analysis
        print("\n1. Sample Submission Analysis")
        sample_sub = pd.read_csv('sample_submission.csv')
        print(f"Shape: {sample_sub.shape}")
        print(f"Columns: {list(sample_sub.columns)}")
        print("First 5 rows:")
        print(sample_sub.head().to_string())

        # Train labels analysis
        print("\n2. Train Labels Analysis")
        train_labels = pd.read_csv('train_labels.csv')
        print(f"Shape: {train_labels.shape}")
        print(f"Columns: {list(train_labels.columns)}")
        print("First 5 rows:")
        print(train_labels.head().to_string())

        print("\nTarget Distribution:")
        target_counts = train_labels['target'].value_counts()
        print(target_counts.to_string())
        print(f"Target rate (default rate): {train_labels['target'].mean():.4f}")

        # Train data sample analysis
        print("\n3. Train Data Sample Analysis")
        train_sample = pd.read_csv('train_data.csv', nrows=1000)
        print(f"Sample shape: {train_sample.shape}")
        print(f"Total columns: {len(train_sample.columns)}")
        print(f"First 10 columns: {list(train_sample.columns[:10])}")

        # Check data types
        print("\nData types summary:")
        dtype_counts = train_sample.dtypes.value_counts()
        print(dtype_counts.to_string())

        # Check for key columns
        if 'customer_ID' in train_sample.columns:
            print(f"\nUnique customers in sample: {train_sample['customer_ID'].nunique()}")
            print(f"Total rows per customer (avg): {len(train_sample) / train_sample['customer_ID'].nunique():.2f}")

        if 'S_2' in train_sample.columns:
            print(f"\nS_2 column (statement date):")
            print(f"Type: {train_sample['S_2'].dtype}")
            print(f"Sample values: {train_sample['S_2'].head().tolist()}")
            print(f"Unique dates: {train_sample['S_2'].nunique()}")

        # Missing values analysis
        print("\n4. Missing Values Analysis (sample)")
        missing_counts = train_sample.isnull().sum()
        missing_pct = (missing_counts / len(train_sample)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_counts,
            'Missing_Percentage': missing_pct
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
        print(f"Columns with missing values: {len(missing_df)}")
        if len(missing_df) > 0:
            print("Top 10 columns with most missing values:")
            print(missing_df.head(10).to_string())

        # Numeric columns analysis
        print("\n5. Numeric Columns Analysis")
        numeric_cols = train_sample.select_dtypes(include=[np.number]).columns
        print(f"Number of numeric columns: {len(numeric_cols)}")
        print(f"Numeric columns (first 10): {list(numeric_cols[:10])}")

        # Categorical columns analysis
        print("\n6. Categorical Columns Analysis")
        categorical_cols = train_sample.select_dtypes(include=['object']).columns
        print(f"Number of categorical columns: {len(categorical_cols)}")
        print(f"Categorical columns: {list(categorical_cols)}")

        for col in categorical_cols:
            unique_count = train_sample[col].nunique()
            print(f"{col}: {unique_count} unique values")
            if unique_count <= 10:
                print(f"  Values: {train_sample[col].unique().tolist()}")

        print("\n=== Analysis Completed Successfully ===")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

# 標準出力を元に戻す
sys.stdout = sys.__stdout__
print("Analysis completed. Results saved to analysis_results.txt")
