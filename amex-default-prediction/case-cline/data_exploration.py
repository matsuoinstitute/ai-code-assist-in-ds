import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=== Data Exploration ===")

# File sizes
import os
print("File sizes:")
for file in ['train_data.csv', 'test_data.csv', 'train_labels.csv', 'sample_submission.csv']:
    if os.path.exists(file):
        size = os.path.getsize(file) / (1024**3)
        print(f'{file}: {size:.2f} GB')

# Load small samples first
print("\n=== Loading sample data ===")
train_sample = pd.read_csv('train_data.csv', nrows=1000)
print(f"Train data sample shape: {train_sample.shape}")
print(f"Train data columns: {list(train_sample.columns)}")

labels_sample = pd.read_csv('train_labels.csv', nrows=1000)
print(f"Labels sample shape: {labels_sample.shape}")
print(f"Labels columns: {list(labels_sample.columns)}")

test_sample = pd.read_csv('test_data.csv', nrows=1000)
print(f"Test data sample shape: {test_sample.shape}")

sample_sub = pd.read_csv('sample_submission.csv')
print(f"Sample submission shape: {sample_sub.shape}")
print(f"Sample submission columns: {list(sample_sub.columns)}")

print("\n=== Data Preview ===")
print("Train data sample:")
print(train_sample.head())

print("\nLabels sample:")
print(labels_sample.head())

print("\nSample submission:")
print(sample_sub.head())

# Check for missing values
print("\n=== Missing Values Analysis ===")
print("Train data missing values:")
missing_train = train_sample.isnull().sum()
print(missing_train[missing_train > 0])

print("\nLabels missing values:")
missing_labels = labels_sample.isnull().sum()
print(missing_labels[missing_labels > 0])

# Data types
print("\n=== Data Types ===")
print("Train data types:")
print(train_sample.dtypes.value_counts())

print("\nLabels data types:")
print(labels_sample.dtypes.value_counts())

# Unique values in categorical columns
print("\n=== Categorical Analysis ===")
categorical_cols = train_sample.select_dtypes(include=['object']).columns
print(f"Categorical columns: {list(categorical_cols)}")

for col in categorical_cols[:5]:  # Show first 5 categorical columns
    print(f"{col}: {train_sample[col].nunique()} unique values")
    if train_sample[col].nunique() < 20:
        print(f"  Values: {train_sample[col].unique()}")

# Target distribution
print("\n=== Target Distribution ===")
if 'target' in labels_sample.columns:
    target_dist = labels_sample['target'].value_counts()
    print(target_dist)
    print(f"Target rate: {labels_sample['target'].mean():.4f}")
