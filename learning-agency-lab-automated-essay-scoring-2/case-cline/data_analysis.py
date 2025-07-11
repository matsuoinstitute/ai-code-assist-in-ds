import pandas as pd

# Analyze data and write to file
with open('data_info.txt', 'w') as f:
    # Train data analysis
    f.write('=== TRAIN DATA INFO ===\n')
    train = pd.read_csv('train.csv')
    f.write(f'Train shape: {train.shape}\n')
    f.write(f'Train columns: {list(train.columns)}\n')
    f.write('\nFirst few rows:\n')
    f.write(str(train.head()) + '\n')
    f.write('\nData types:\n')
    f.write(str(train.dtypes) + '\n')

    # Check for score columns
    score_columns = [col for col in train.columns if 'score' in col.lower()]
    if score_columns:
        f.write(f'\nScore columns found: {score_columns}\n')
        for col in score_columns:
            f.write(f'\n{col} stats:\n')
            f.write(str(train[col].describe()) + '\n')
            f.write(f'{col} unique values: {sorted(train[col].unique())}\n')

    # Test data analysis
    f.write('\n=== TEST DATA INFO ===\n')
    test = pd.read_csv('test.csv')
    f.write(f'Test shape: {test.shape}\n')
    f.write(f'Test columns: {list(test.columns)}\n')
    f.write('\nFirst few rows:\n')
    f.write(str(test.head()) + '\n')

    # Missing values
    f.write('\n=== MISSING VALUES ===\n')
    f.write('Train missing values:\n')
    f.write(str(train.isnull().sum()) + '\n')
    f.write('\nTest missing values:\n')
    f.write(str(test.isnull().sum()) + '\n')

print("Analysis complete. Check data_info.txt")
