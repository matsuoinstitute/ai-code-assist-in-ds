import subprocess
import time
import os
import sys

def run_with_monitoring():
    print("Starting advanced model training...")
    print("This may take several minutes...")

    # プロセスを開始
    process = subprocess.Popen(
        ['python3', 'advanced_amex_model.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    # リアルタイムで出力を表示
    output_lines = []
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            output_lines.append(output.strip())

    # プロセス終了を待つ
    return_code = process.poll()

    # 結果をファイルに保存
    with open('monitoring_output.txt', 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"\nProcess completed with return code: {return_code}")

    # 生成されたファイルを確認
    print("\nChecking generated files...")
    files = os.listdir('.')
    csv_files = [f for f in files if f.endswith('.csv')]
    print(f"CSV files: {csv_files}")

    if 'advanced_amex_submission.csv' in csv_files:
        print("✓ Submission file created successfully!")
        # ファイルサイズを確認
        size = os.path.getsize('advanced_amex_submission.csv')
        print(f"File size: {size} bytes")

        # 最初の数行を表示
        import pandas as pd
        try:
            df = pd.read_csv('advanced_amex_submission.csv')
            print(f"Submission shape: {df.shape}")
            print("First 5 rows:")
            print(df.head())
            print(f"Prediction range: [{df['prediction'].min():.6f}, {df['prediction'].max():.6f}]")
        except Exception as e:
            print(f"Error reading submission file: {e}")
    else:
        print("⚠ Submission file not found")

if __name__ == "__main__":
    run_with_monitoring()
