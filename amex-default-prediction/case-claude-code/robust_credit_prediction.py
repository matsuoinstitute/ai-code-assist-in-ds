#!/usr/bin/env python3
"""
Robust Credit Default Prediction Model
Handles mixed data types and large-scale datasets
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import gc
from typing import List, Tuple, Dict, Any
from pathlib import Path

# ML Libraries
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

class RobustCreditPredictor:
    """
    Robust credit default prediction system for mixed data types
    """
    
    def __init__(self, data_path: str = "."):
        self.data_path = Path(data_path)
        self.label_encoders = {}
        
    def analyze_data_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Analyze and categorize columns by data type"""
        numeric_cols = []
        categorical_cols = []
        
        for col in df.columns:
            if col == 'customer_ID':
                continue
                
            if df[col].dtype in ['int64', 'float64']:
                numeric_cols.append(col)
            else:
                # Check if it's a categorical column
                unique_values = df[col].nunique()
                if unique_values < 1000:  # Arbitrary threshold for categorical
                    categorical_cols.append(col)
                else:
                    # Handle string-like columns
                    if col.startswith('D_'):  # Delinquency columns might be categorical
                        categorical_cols.append(col)
        
        return {
            'numeric': numeric_cols,
            'categorical': categorical_cols
        }
    
    def create_customer_features_robust(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer-level features handling mixed data types
        """
        # Analyze data types
        col_types = self.analyze_data_types(df)
        
        print(f"Found {len(col_types['numeric'])} numeric and {len(col_types['categorical'])} categorical columns")
        
        # Get last observation per customer
        customer_last = df.groupby('customer_ID').last().reset_index()
        
        # Process numeric columns
        numeric_features = {}
        if col_types['numeric']:
            for col in col_types['numeric']:
                if col in df.columns:
                    grouped = df.groupby('customer_ID')[col]
                    numeric_features[f'{col}_mean'] = grouped.mean()
                    numeric_features[f'{col}_std'] = grouped.std()
                    numeric_features[f'{col}_min'] = grouped.min()
                    numeric_features[f'{col}_max'] = grouped.max()
                    numeric_features[f'{col}_last'] = grouped.last()
        
        # Convert to DataFrame
        if numeric_features:
            numeric_df = pd.DataFrame(numeric_features).reset_index()
        else:
            numeric_df = pd.DataFrame({'customer_ID': df['customer_ID'].unique()})
        
        # Process categorical columns
        categorical_features = {}
        if col_types['categorical']:
            for col in col_types['categorical']:
                if col in df.columns:
                    grouped = df.groupby('customer_ID')[col]
                    # Mode (most frequent value) for categorical
                    categorical_features[f'{col}_mode'] = grouped.agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'missing')
                    # Count of unique values
                    categorical_features[f'{col}_nunique'] = grouped.nunique()
        
        # Convert to DataFrame
        if categorical_features:
            categorical_df = pd.DataFrame(categorical_features).reset_index()
        else:
            categorical_df = pd.DataFrame({'customer_ID': df['customer_ID'].unique()})
        
        # Merge all features
        customer_features = customer_last.merge(numeric_df, on='customer_ID', how='left')
        customer_features = customer_features.merge(categorical_df, on='customer_ID', how='left')
        
        # Advanced engineered features
        # Risk and balance features
        balance_cols = [col for col in customer_features.columns if col.startswith('B_') and '_mean' in col]
        if balance_cols:
            customer_features['total_balance_mean'] = customer_features[balance_cols].sum(axis=1)
            customer_features['max_balance_mean'] = customer_features[balance_cols].max(axis=1)
        
        # Spending features
        spend_cols = [col for col in customer_features.columns if col.startswith('S_') and '_mean' in col]
        if spend_cols:
            customer_features['total_spending_mean'] = customer_features[spend_cols].sum(axis=1)
        
        # Risk features
        risk_cols = [col for col in customer_features.columns if col.startswith('R_') and '_mean' in col]
        if risk_cols:
            customer_features['total_risk_mean'] = customer_features[risk_cols].sum(axis=1)
        
        return customer_features
    
    def load_and_process_data(self, sample_size: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and process data with optional sampling for memory efficiency"""
        
        print("Loading train labels...")
        train_labels = pd.read_csv(self.data_path / "train_labels.csv")
        
        # For large datasets, we can sample customers
        if sample_size:
            sampled_customers = train_labels.sample(n=min(sample_size, len(train_labels)), random_state=42)['customer_ID']
            print(f"Sampling {len(sampled_customers)} customers for training")
        else:
            sampled_customers = train_labels['customer_ID']
        
        print("Loading and processing train data...")
        # Process train data in chunks
        train_chunks = []
        chunk_size = 100000
        
        for chunk in pd.read_csv(self.data_path / "train_data.csv", chunksize=chunk_size):
            # Filter to sampled customers
            if sample_size:
                chunk = chunk[chunk['customer_ID'].isin(sampled_customers)]
            
            if len(chunk) > 0:
                processed_chunk = self.create_customer_features_robust(chunk)
                train_chunks.append(processed_chunk)
                
        train_features = pd.concat(train_chunks, ignore_index=True)
        
        print("Loading and processing test data...")
        # Process test data similarly
        test_chunks = []
        for chunk in pd.read_csv(self.data_path / "test_data.csv", chunksize=chunk_size):
            processed_chunk = self.create_customer_features_robust(chunk)
            test_chunks.append(processed_chunk)
            
        test_features = pd.concat(test_chunks, ignore_index=True)
        
        return train_features, train_labels, test_features
    
    def prepare_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare features for modeling"""
        
        # Identify feature columns
        feature_cols = [col for col in train_df.columns if col not in ['customer_ID', 'target']]
        
        # Handle categorical features
        categorical_cols = []
        for col in feature_cols:
            if train_df[col].dtype == 'object':
                categorical_cols.append(col)
                
                # Label encode categorical variables
                le = LabelEncoder()
                combined_values = pd.concat([train_df[col], test_df[col]]).astype(str)
                le.fit(combined_values)
                
                train_df[col] = le.transform(train_df[col].astype(str))
                test_df[col] = le.transform(test_df[col].astype(str))
                
                self.label_encoders[col] = le
        
        # Handle missing values
        for col in feature_cols:
            if train_df[col].isnull().sum() > 0:
                if col in categorical_cols:
                    fill_value = train_df[col].mode()[0] if len(train_df[col].mode()) > 0 else 0
                else:
                    fill_value = train_df[col].median()
                
                train_df[col] = train_df[col].fillna(fill_value)
                test_df[col] = test_df[col].fillna(fill_value)
        
        return train_df[feature_cols], test_df[feature_cols]
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
        """Train ensemble of models"""
        
        print("Training models...")
        
        # Split for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # LightGBM
        print("Training LightGBM...")
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 64,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 100,
            'max_depth': 8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'n_estimators': 300,
            'random_state': 42,
            'n_jobs': -1,
            'force_row_wise': True,
            'verbose': -1
        }
        
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(
            X_train_split, y_train_split,
            eval_set=[(X_val_split, y_val_split)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(30)]
        )
        
        # XGBoost
        print("Training XGBoost...")
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 8,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
            'verbosity': 0
        }
        
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(
            X_train_split, y_train_split,
            eval_set=[(X_val_split, y_val_split)],
            early_stopping_rounds=30,
            verbose=False
        )
        
        # Validation scores
        lgb_val_preds = lgb_model.predict_proba(X_val_split)[:, 1]
        xgb_val_preds = xgb_model.predict_proba(X_val_split)[:, 1]
        
        lgb_auc = roc_auc_score(y_val_split, lgb_val_preds)
        xgb_auc = roc_auc_score(y_val_split, xgb_val_preds)
        
        print(f"LightGBM Validation AUC: {lgb_auc:.6f}")
        print(f"XGBoost Validation AUC: {xgb_auc:.6f}")
        
        # Weighted ensemble based on validation performance
        lgb_weight = lgb_auc / (lgb_auc + xgb_auc)
        xgb_weight = xgb_auc / (lgb_auc + xgb_auc)
        
        # Final predictions
        lgb_test_preds = lgb_model.predict_proba(X_test)[:, 1]
        xgb_test_preds = xgb_model.predict_proba(X_test)[:, 1]
        
        final_preds = lgb_weight * lgb_test_preds + xgb_weight * xgb_test_preds
        
        print(f"Ensemble weights - LightGBM: {lgb_weight:.3f}, XGBoost: {xgb_weight:.3f}")
        
        return final_preds
    
    def run_pipeline(self, sample_size: int = 50000):
        """Run the complete pipeline"""
        
        print("="*70)
        print("ROBUST CREDIT DEFAULT PREDICTION PIPELINE")
        print("="*70)
        
        # Load and process data
        train_features, train_labels, test_features = self.load_and_process_data(sample_size=sample_size)
        
        # Merge train features with labels
        train_final = train_features.merge(train_labels, on='customer_ID', how='inner')
        
        # Prepare features
        X_train, X_test = self.prepare_features(train_final.copy(), test_features.copy())
        y_train = train_final['target']
        
        print(f"Final feature shape - Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"Target distribution: {y_train.value_counts(normalize=True).to_dict()}")
        
        # Train models
        predictions = self.train_models(X_train, y_train, X_test)
        
        # Create submission
        sample_submission = pd.read_csv(self.data_path / "sample_submission.csv")
        
        submission = pd.DataFrame({
            'customer_ID': sample_submission['customer_ID'],
            'prediction': predictions
        })
        
        # Save results
        submission.to_csv('submission.csv', index=False)
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Submission saved to: submission.csv")
        print(f"Submission shape: {submission.shape}")
        print(f"Prediction statistics:")
        print(f"  Mean: {predictions.mean():.6f}")
        print(f"  Std: {predictions.std():.6f}")
        print(f"  Min: {predictions.min():.6f}")
        print(f"  Max: {predictions.max():.6f}")
        
        return submission

if __name__ == "__main__":
    # Run with a reasonable sample size for memory efficiency
    predictor = RobustCreditPredictor()
    submission = predictor.run_pipeline(sample_size=100000)  # Use 100k customers for training