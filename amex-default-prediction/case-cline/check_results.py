import os
import pandas as pd

print("=== Checking Results ===")

# Check if files exist
files_to_check = [
    'simple_submission.csv',
    'advanced_amex_submission.csv',
    'simple_test_output.txt',
    'advanced_model_output.txt'
]

print("File existence check:")
for file in files_to_check:
    exists = os.path.exists(file)
    if exists:
        size = os.path.getsize(file)
        print(f"✓ {file}: {size} bytes")
    else:
        print(f"✗ {file}: Not found")

# Check CSV files
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
print(f"\nAll CSV files: {csv_files}")

# Try to read submission files
for csv_file in ['simple_submission.csv', 'advanced_amex_submission.csv']:
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            print(f"\n{csv_file}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            if 'prediction' in df.columns:
                print(f"  Prediction range: [{df['prediction'].min():.6f}, {df['prediction'].max():.6f}]")
                print(f"  Mean prediction: {df['prediction'].mean():.6f}")
            print("  First 5 rows:")
            print(df.head().to_string())
        except Exception as e:
            print(f"  Error reading {csv_file}: {e}")

# Try to read output files
for txt_file in ['simple_test_output.txt', 'advanced_model_output.txt']:
    if os.path.exists(txt_file):
        try:
            with open(txt_file, 'r') as f:
                content = f.read()
            print(f"\n{txt_file} (last 500 chars):")
            print(content[-500:] if len(content) > 500 else content)
        except Exception as e:
            print(f"  Error reading {txt_file}: {e}")

print("\n=== Check completed ===")
