#!/usr/bin/env python3
"""
Optimized Credit Default Prediction Model
Memory-efficient industrial-scale machine learning pipeline
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import gc
import pickle
from typing import List, Tuple, Dict, Any
from pathlib import Path

# ML Libraries
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Data processing
from tqdm import tqdm

class OptimizedCreditPredictor:
    """
    Memory-efficient credit default prediction system
    """
    
    def __init__(self, data_path: str = "."):
        self.data_path = Path(data_path)
        
    def load_data_chunked(self, chunksize: int = 50000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and process data in chunks to manage memory"""
        print("Loading datasets in chunks...")
        
        # Load labels first (smaller file)
        train_labels = pd.read_csv(self.data_path / "train_labels.csv")
        print(f"Train labels shape: {train_labels.shape}")
        
        # For the large training data, we'll process by customer
        print("Processing training data...")
        train_chunks = []
        chunk_count = 0
        
        for chunk in pd.read_csv(self.data_path / "train_data.csv", chunksize=chunksize):
            # Process each chunk
            processed_chunk = self.create_customer_features(chunk)
            train_chunks.append(processed_chunk)
            chunk_count += 1
            
            if chunk_count % 10 == 0:
                print(f"Processed {chunk_count} chunks, latest chunk shape: {chunk.shape}")
                gc.collect()
        
        # Combine all processed chunks
        train_features = pd.concat(train_chunks, ignore_index=True)
        
        # Similarly for test data
        print("Processing test data...")
        test_chunks = []
        chunk_count = 0
        
        for chunk in pd.read_csv(self.data_path / "test_data.csv", chunksize=chunksize):
            processed_chunk = self.create_customer_features(chunk)
            test_chunks.append(processed_chunk)
            chunk_count += 1
            
            if chunk_count % 10 == 0:
                print(f"Processed {chunk_count} test chunks")
                gc.collect()
        
        test_features = pd.concat(test_chunks, ignore_index=True)
        
        print(f"Final train features shape: {train_features.shape}")
        print(f"Final test features shape: {test_features.shape}")
        
        return train_features, train_labels, test_features
    
    def create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer-level features from time series data
        """
        # Get last observation per customer (most recent)
        customer_last = df.groupby('customer_ID').last().reset_index()
        
        # Statistical aggregations
        customer_stats = df.groupby('customer_ID').agg({
            # Balance features (B_)
            **{col: ['mean', 'std', 'min', 'max'] for col in df.columns if col.startswith('B_')},
            # Spend features (S_)  
            **{col: ['mean', 'std', 'sum', 'max'] for col in df.columns if col.startswith('S_') and col != 'S_2'},
            # Payment features (P_)
            **{col: ['mean', 'std', 'max'] for col in df.columns if col.startswith('P_')},
            # Delinquency features (D_)
            **{col: ['mean', 'max', 'sum'] for col in df.columns if col.startswith('D_')},
            # Risk features (R_)
            **{col: ['mean', 'max', 'sum'] for col in df.columns if col.startswith('R_')}
        }).reset_index()
        
        # Flatten column names
        customer_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in customer_stats.columns]
        
        # Merge last observation with stats
        customer_features = customer_last.merge(customer_stats, on='customer_ID', how='left')
        
        # Create advanced features
        if 'B_1' in df.columns and 'B_2' in df.columns:
            customer_features['credit_utilization'] = customer_features['B_1'] / (customer_features['B_2'] + 1e-8)
        
        # Risk concentration
        risk_cols = [col for col in customer_features.columns if col.startswith('R_') and '_' not in col[2:]]
        if risk_cols:
            customer_features['total_risk'] = customer_features[risk_cols].sum(axis=1)
            customer_features['max_risk'] = customer_features[risk_cols].max(axis=1)
        
        # Balance volatility
        balance_std_cols = [col for col in customer_features.columns if col.startswith('B_') and col.endswith('_std')]
        if balance_std_cols:
            customer_features['balance_volatility'] = customer_features[balance_std_cols].mean(axis=1)
        
        return customer_features
    
    def prepare_model_data(self, train_features: pd.DataFrame, train_labels: pd.DataFrame, 
                          test_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Prepare final datasets for modeling"""
        
        # Merge train features with labels
        train_final = train_features.merge(train_labels, on='customer_ID', how='inner')
        
        # Feature selection - remove ID and date columns
        feature_cols = [col for col in train_final.columns 
                       if col not in ['customer_ID', 'target', 'S_2']]
        
        X_train = train_final[feature_cols]
        y_train = train_final['target']
        X_test = test_features[feature_cols]
        
        # Handle missing values
        print("Handling missing values...")
        for col in X_train.columns:
            if X_train[col].dtype in ['float64', 'int64']:
                fill_value = X_train[col].median()
                X_train[col] = X_train[col].fillna(fill_value)
                X_test[col] = X_test[col].fillna(fill_value)
            else:
                X_train[col] = X_train[col].fillna('missing')
                X_test[col] = X_test[col].fillna('missing')
        
        print(f"Final training shape: {X_train.shape}")
        print(f"Final test shape: {X_test.shape}")
        print(f"Target distribution: {y_train.value_counts(normalize=True)}")
        
        return X_train, y_train, X_test
    
    def train_lightgbm_optimized(self, X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMClassifier:
        """Train optimized LightGBM model"""
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 64,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 200,
            'max_depth': 8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'n_estimators': 500,
            'random_state': 42,
            'n_jobs': -1,
            'force_row_wise': True
        }
        
        model = lgb.LGBMClassifier(**params)
        
        # Split for validation
        split_idx = int(0.8 * len(X_train))
        X_train_split, X_val_split = X_train[:split_idx], X_train[split_idx:]
        y_train_split, y_val_split = y_train[:split_idx], y_train[split_idx:]
        
        model.fit(
            X_train_split, y_train_split,
            eval_set=[(X_val_split, y_val_split)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )
        
        # Validation score
        val_preds = model.predict_proba(X_val_split)[:, 1]
        val_auc = roc_auc_score(y_val_split, val_preds)
        print(f"LightGBM Validation AUC: {val_auc:.6f}")
        
        return model
    
    def train_xgboost_optimized(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
        """Train optimized XGBoost model"""
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 8,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'
        }
        
        model = xgb.XGBClassifier(**params)
        
        # Split for validation
        split_idx = int(0.8 * len(X_train))
        X_train_split, X_val_split = X_train[:split_idx], X_train[split_idx:]
        y_train_split, y_val_split = y_train[:split_idx], y_train[split_idx:]
        
        model.fit(
            X_train_split, y_train_split,
            eval_set=[(X_val_split, y_val_split)],
            early_stopping_rounds=50,
            verbose=50
        )
        
        # Validation score
        val_preds = model.predict_proba(X_val_split)[:, 1]
        val_auc = roc_auc_score(y_val_split, val_preds)
        print(f"XGBoost Validation AUC: {val_auc:.6f}")
        
        return model
    
    def create_simple_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series, 
                              X_test: pd.DataFrame) -> np.ndarray:
        """Create a simple but effective ensemble"""
        
        print("Training ensemble models...")
        
        # Train individual models
        print("\nTraining LightGBM...")
        lgb_model = self.train_lightgbm_optimized(X_train, y_train)
        
        print("\nTraining XGBoost...")
        xgb_model = self.train_xgboost_optimized(X_train, y_train)
        
        # Make predictions
        print("\nMaking predictions...")
        lgb_preds = lgb_model.predict_proba(X_test)[:, 1]
        xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
        
        # Simple average ensemble
        ensemble_preds = (lgb_preds + xgb_preds) / 2
        
        print(f"LightGBM predictions - Mean: {lgb_preds.mean():.4f}, Std: {lgb_preds.std():.4f}")
        print(f"XGBoost predictions - Mean: {xgb_preds.mean():.4f}, Std: {xgb_preds.std():.4f}")
        print(f"Ensemble predictions - Mean: {ensemble_preds.mean():.4f}, Std: {ensemble_preds.std():.4f}")
        
        return ensemble_preds
    
    def run_optimized_pipeline(self):
        """Run the optimized machine learning pipeline"""
        
        print("="*60)
        print("OPTIMIZED CREDIT DEFAULT PREDICTION PIPELINE")
        print("="*60)
        
        # 1. Load and process data in chunks
        train_features, train_labels, test_features = self.load_data_chunked()
        
        # 2. Prepare model data
        X_train, y_train, X_test = self.prepare_model_data(train_features, train_labels, test_features)
        
        # Clear memory
        del train_features, test_features
        gc.collect()
        
        # 3. Train ensemble
        ensemble_preds = self.create_simple_ensemble(X_train, y_train, X_test)
        
        # 4. Create submission
        sample_submission = pd.read_csv(self.data_path / "sample_submission.csv")
        
        submission = pd.DataFrame({
            'customer_ID': sample_submission['customer_ID'],
            'prediction': ensemble_preds
        })
        
        # 5. Save results
        submission.to_csv('submission.csv', index=False)
        
        print("\n" + "="*60)
        print("OPTIMIZED PIPELINE COMPLETED")
        print("="*60)
        print(f"Submission saved to: submission.csv")
        print(f"Submission shape: {submission.shape}")
        print(f"Prediction statistics:")
        print(f"  Mean: {ensemble_preds.mean():.6f}")
        print(f"  Std: {ensemble_preds.std():.6f}")
        print(f"  Min: {ensemble_preds.min():.6f}")
        print(f"  Max: {ensemble_preds.max():.6f}")
        print(f"  25th percentile: {np.percentile(ensemble_preds, 25):.6f}")
        print(f"  50th percentile: {np.percentile(ensemble_preds, 50):.6f}")
        print(f"  75th percentile: {np.percentile(ensemble_preds, 75):.6f}")
        
        return submission

if __name__ == "__main__":
    # Initialize and run the optimized predictor
    predictor = OptimizedCreditPredictor()
    submission = predictor.run_optimized_pipeline()