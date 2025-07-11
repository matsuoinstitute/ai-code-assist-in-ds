import pandas as pd
import numpy as np

# Load and examine train data
print('=== TRAIN DATA INFO ===')
train = pd.read_csv('train.csv')
print(f'Train shape: {train.shape}')
print(f'Train columns: {list(train.columns)}')
print('\nFirst few rows:')
print(train.head())
print('\nData types:')
print(train.dtypes)

# Check for score column
score_columns = [col for col in train.columns if 'score' in col.lower()]
if score_columns:
    print(f'\nScore columns found: {score_columns}')
    for col in score_columns:
        print(f'\n{col} stats:')
        print(train[col].describe())
        print(f'{col} unique values: {sorted(train[col].unique())}')

print('\n=== TEST DATA INFO ===')
test = pd.read_csv('test.csv')
print(f'Test shape: {test.shape}')
print(f'Test columns: {list(test.columns)}')
print('\nFirst few rows:')
print(test.head())

# Check for missing values
print('\n=== MISSING VALUES ===')
print('Train missing values:')
print(train.isnull().sum())
print('\nTest missing values:')
print(test.isnull().sum())
