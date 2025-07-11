"""
Simple model test to verify the pipeline works
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import sys

def simple_test():
    try:
        print("=== Simple Model Test ===")

        # Load data
        print("Loading data...")
        train_labels = pd.read_csv('train_labels.csv')
        sample_sub = pd.read_csv('sample_submission.csv')

        # Load small sample
        train_data = pd.read_csv('train_data.csv', nrows=5000)
        test_data = pd.read_csv('test_data.csv', nrows=2500)

        print(f"Train data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        print(f"Target rate: {train_labels['target'].mean():.4f}")

        # Simple feature engineering
        print("Simple feature engineering...")

        def create_simple_features(df):
            features = []
            for customer_id, group in df.groupby('customer_ID'):
                feature_dict = {'customer_ID': customer_id}

                # Basic stats for numeric columns
                numeric_cols = group.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col != 'customer_ID':
                        values = group[col].dropna()
                        if len(values) > 0:
                            feature_dict[f'{col}_mean'] = values.mean()
                            feature_dict[f'{col}_last'] = values.iloc[-1]
                            feature_dict[f'{col}_std'] = values.std()

                # Number of statements
                feature_dict['num_statements'] = len(group)

                features.append(feature_dict)

            return pd.DataFrame(features)

        # Create features
        train_features = create_simple_features(train_data)
        test_features = create_simple_features(test_data)

        print(f"Train features shape: {train_features.shape}")
        print(f"Test features shape: {test_features.shape}")

        # Merge with labels
        train_features = train_features.merge(train_labels, on='customer_ID', how='left')

        # Prepare data
        feature_cols = [col for col in train_features.columns if col not in ['customer_ID', 'target']]
        X = train_features[feature_cols].fillna(0)
        y = train_features['target']

        print(f"Final feature shape: {X.shape}")
        print(f"Number of features: {len(feature_cols)}")

        # Train simple model
        print("Training Random Forest...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # Validate
        val_pred = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, val_pred)
        print(f"Validation AUC: {auc_score:.6f}")

        # Predict on test
        print("Making predictions...")
        X_test = test_features[feature_cols].fillna(0)
        test_pred = model.predict_proba(X_test)[:, 1]

        # Create submission
        submission = pd.DataFrame({
            'customer_ID': test_features['customer_ID'],
            'prediction': test_pred
        })

        # Merge with sample submission format
        final_submission = sample_sub.merge(submission, on='customer_ID', how='left')
        final_submission['prediction'] = final_submission['prediction_y'].fillna(0.5)
        final_submission = final_submission[['customer_ID', 'prediction']]

        # Save
        final_submission.to_csv('simple_submission.csv', index=False)

        print(f"Submission created: simple_submission.csv")
        print(f"Submission shape: {final_submission.shape}")
        print(f"Prediction range: [{final_submission['prediction'].min():.6f}, {final_submission['prediction'].max():.6f}]")

        # Show sample
        print("\nSample predictions:")
        print(final_submission.head(10))

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_test()
    if success:
        print("\n✓ Simple model test completed successfully!")
    else:
        print("\n✗ Simple model test failed!")
