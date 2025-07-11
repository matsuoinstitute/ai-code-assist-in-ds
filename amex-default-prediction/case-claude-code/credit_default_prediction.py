#!/usr/bin/env python3
"""
Credit Default Prediction Model
Industrial-scale machine learning pipeline for credit default prediction
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
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Data processing
import polars as pl
from tqdm import tqdm

class CreditDefaultPredictor:
    """
    Industrial-scale credit default prediction system
    """
    
    def __init__(self, data_path: str = "."):
        self.data_path = Path(data_path)
        self.models = {}
        self.feature_importance = {}
        self.scalers = {}
        
    def load_data(self, use_polars: bool = True) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Load large-scale datasets efficiently using Polars"""
        print("Loading datasets...")
        
        if use_polars:
            # Use Polars for efficient memory usage with large datasets
            train_data = pl.read_csv(self.data_path / "train_data.csv")
            train_labels = pl.read_csv(self.data_path / "train_labels.csv")
            test_data = pl.read_csv(self.data_path / "test_data.csv")
        else:
            # Fallback to pandas with chunking for very large files
            train_data = pd.read_csv(self.data_path / "train_data.csv")
            train_labels = pd.read_csv(self.data_path / "train_labels.csv")
            test_data = pd.read_csv(self.data_path / "test_data.csv")
            
        print(f"Train data shape: {train_data.shape}")
        print(f"Train labels shape: {train_labels.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        return train_data, train_labels, test_data
    
    def create_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Comprehensive feature engineering for credit default prediction
        """
        print("Creating features...")
        
        # Convert to pandas for complex feature engineering
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()
            
        # Identify feature types based on prefixes
        balance_features = [col for col in df.columns if col.startswith('B_')]
        spend_features = [col for col in df.columns if col.startswith('S_')]
        payment_features = [col for col in df.columns if col.startswith('P_')]
        delinquency_features = [col for col in df.columns if col.startswith('D_')]
        risk_features = [col for col in df.columns if col.startswith('R_')]
        
        # 1. Statistical aggregations for each customer's time series
        customer_features = []
        
        # Group by customer and create aggregated features
        grouped = df.groupby('customer_ID')
        
        # Balance features aggregations
        for feature in balance_features:
            if feature in df.columns:
                df[f'{feature}_mean'] = grouped[feature].transform('mean')
                df[f'{feature}_std'] = grouped[feature].transform('std')
                df[f'{feature}_max'] = grouped[feature].transform('max')
                df[f'{feature}_min'] = grouped[feature].transform('min')
                df[f'{feature}_last'] = grouped[feature].transform('last')
                df[f'{feature}_first'] = grouped[feature].transform('first')
                df[f'{feature}_trend'] = df[f'{feature}_last'] - df[f'{feature}_first']
                
        # Spend features aggregations  
        for feature in spend_features:
            if feature in df.columns:
                df[f'{feature}_mean'] = grouped[feature].transform('mean')
                df[f'{feature}_std'] = grouped[feature].transform('std')
                df[f'{feature}_max'] = grouped[feature].transform('max')
                df[f'{feature}_sum'] = grouped[feature].transform('sum')
                df[f'{feature}_last'] = grouped[feature].transform('last')
                
        # Payment features aggregations
        for feature in payment_features:
            if feature in df.columns:
                df[f'{feature}_mean'] = grouped[feature].transform('mean')
                df[f'{feature}_std'] = grouped[feature].transform('std')
                df[f'{feature}_last'] = grouped[feature].transform('last')
                
        # Risk and delinquency features
        for feature in risk_features + delinquency_features:
            if feature in df.columns:
                df[f'{feature}_mean'] = grouped[feature].transform('mean')
                df[f'{feature}_max'] = grouped[feature].transform('max')
                df[f'{feature}_last'] = grouped[feature].transform('last')
                df[f'{feature}_sum'] = grouped[feature].transform('sum')
        
        # 2. Advanced engineered features
        
        # Credit utilization metrics
        if 'B_1' in df.columns and 'B_2' in df.columns:
            df['credit_utilization'] = df['B_1'] / (df['B_2'] + 1e-8)
            df['credit_utilization_mean'] = grouped['credit_utilization'].transform('mean')
            
        # Payment behavior trends
        if 'P_2' in df.columns:
            df['payment_trend'] = grouped['P_2'].transform(lambda x: x.diff().mean())
            
        # Balance volatility
        balance_cols = [col for col in df.columns if col.startswith('B_') and '_' not in col[2:]]
        if balance_cols:
            df['balance_volatility'] = df[balance_cols].std(axis=1)
            
        # Spending consistency
        spend_cols = [col for col in df.columns if col.startswith('S_') and '_' not in col[2:]]
        if spend_cols:
            df['spending_consistency'] = 1 / (df[spend_cols].std(axis=1) + 1e-8)
            
        # Risk concentration
        risk_cols = [col for col in df.columns if col.startswith('R_') and '_' not in col[2:]]
        if risk_cols:
            df['risk_concentration'] = df[risk_cols].max(axis=1)
            
        # 3. Time-based features (assuming S_2 contains dates)
        if 'S_2' in df.columns:
            df['S_2'] = pd.to_datetime(df['S_2'])
            df['month'] = df['S_2'].dt.month
            df['quarter'] = df['S_2'].dt.quarter
            df['day_of_month'] = df['S_2'].dt.day
            
            # Recency features
            df['days_since_last'] = (df['S_2'].max() - df['S_2']).dt.days
            
        # 4. Customer-level aggregations (take last observation per customer)
        customer_df = df.groupby('customer_ID').last().reset_index()
        
        print(f"Features created. Shape: {customer_df.shape}")
        return customer_df
    
    def prepare_datasets(self, train_df: pl.DataFrame, train_labels: pl.DataFrame, 
                        test_df: pl.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Prepare training and test datasets with features"""
        
        # Create features for train and test
        train_features = self.create_features(train_df)
        test_features = self.create_features(test_df)
        
        # Convert labels to pandas if needed
        if isinstance(train_labels, pl.DataFrame):
            train_labels = train_labels.to_pandas()
            
        # Merge with labels
        train_final = train_features.merge(train_labels, on='customer_ID', how='inner')
        
        # Separate features and target
        feature_cols = [col for col in train_final.columns 
                       if col not in ['customer_ID', 'target', 'S_2']]
        
        X_train = train_final[feature_cols]
        y_train = train_final['target']
        X_test = test_features[feature_cols]
        
        # Handle missing values
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())
        
        print(f"Final training shape: {X_train.shape}")
        print(f"Final test shape: {X_test.shape}")
        print(f"Target distribution: {y_train.value_counts(normalize=True)}")
        
        return X_train, y_train, X_test
    
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series, 
                      params: Dict = None) -> lgb.LGBMClassifier:
        """Train LightGBM model with optimal parameters for credit scoring"""
        
        if params is None:
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 100,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 100,
                'max_depth': -1,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'n_estimators': 1000,
                'random_state': 42,
                'n_jobs': -1,
                'importance_type': 'gain'
            }
        
        model = lgb.LGBMClassifier(**params)
        
        # Use early stopping with validation split
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            eval_metric='auc',
            early_stopping_rounds=100,
            verbose=100
        )
        
        return model
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     params: Dict = None) -> xgb.XGBClassifier:
        """Train XGBoost model"""
        
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 1000,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist'
            }
            
        model = xgb.XGBClassifier(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            early_stopping_rounds=100,
            verbose=100
        )
        
        return model
    
    def train_catboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                      params: Dict = None) -> CatBoostClassifier:
        """Train CatBoost model"""
        
        if params is None:
            params = {
                'objective': 'Logloss',
                'eval_metric': 'AUC',
                'depth': 6,
                'learning_rate': 0.05,
                'iterations': 1000,
                'l2_leaf_reg': 3,
                'random_seed': 42,
                'thread_count': -1,
                'early_stopping_rounds': 100,
                'verbose': 100
            }
            
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)
        
        return model
    
    def create_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series, 
                       X_test: pd.DataFrame, cv_folds: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Create ensemble of models with cross-validation"""
        
        print("Training ensemble models...")
        
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Store out-of-fold predictions
        oof_preds = np.zeros(len(X_train))
        test_preds = np.zeros(len(X_test))
        
        # Model weights based on validation performance
        model_weights = []
        
        models_to_train = [
            ('lightgbm', self.train_lightgbm),
            ('xgboost', self.train_xgboost), 
            ('catboost', self.train_catboost)
        ]
        
        ensemble_preds = np.zeros((len(models_to_train), len(X_test)))
        
        for model_idx, (model_name, train_func) in enumerate(models_to_train):
            print(f"\nTraining {model_name}...")
            
            fold_preds = np.zeros(len(X_test))
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                print(f"Fold {fold + 1}/{cv_folds}")
                
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Train model
                model = train_func(X_fold_train, y_fold_train)
                
                # Validate
                val_preds = model.predict_proba(X_fold_val)[:, 1]
                fold_score = roc_auc_score(y_fold_val, val_preds)
                fold_scores.append(fold_score)
                
                # Store OOF predictions
                if model_idx == 0:  # Only for the first model to avoid overwriting
                    oof_preds[val_idx] = val_preds
                
                # Test predictions
                test_fold_preds = model.predict_proba(X_test)[:, 1]
                fold_preds += test_fold_preds / cv_folds
                
                print(f"Fold {fold + 1} AUC: {fold_score:.6f}")
                
                # Clear memory
                del model
                gc.collect()
            
            ensemble_preds[model_idx] = fold_preds
            avg_score = np.mean(fold_scores)
            model_weights.append(avg_score)
            
            print(f"{model_name} CV AUC: {avg_score:.6f}")
        
        # Weighted ensemble
        model_weights = np.array(model_weights)
        model_weights = model_weights / model_weights.sum()
        
        final_test_preds = np.average(ensemble_preds, axis=0, weights=model_weights)
        
        print(f"\nEnsemble weights: {dict(zip([name for name, _ in models_to_train], model_weights))}")
        print(f"Overall OOF AUC: {roc_auc_score(y_train, oof_preds):.6f}")
        
        return oof_preds, final_test_preds
    
    def run_full_pipeline(self):
        """Run the complete machine learning pipeline"""
        
        print("="*50)
        print("CREDIT DEFAULT PREDICTION PIPELINE")
        print("="*50)
        
        # 1. Load data
        train_data, train_labels, test_data = self.load_data()
        
        # 2. Feature engineering and preparation
        X_train, y_train, X_test = self.prepare_datasets(train_data, train_labels, test_data)
        
        # Clear memory
        del train_data, test_data
        gc.collect()
        
        # 3. Train ensemble
        oof_preds, test_preds = self.create_ensemble(X_train, y_train, X_test)
        
        # 4. Create submission
        test_customer_ids = pd.read_csv(self.data_path / "sample_submission.csv")['customer_ID']
        
        submission = pd.DataFrame({
            'customer_ID': test_customer_ids,
            'prediction': test_preds
        })
        
        # 5. Save results
        submission.to_csv('submission.csv', index=False)
        
        print("\n" + "="*50)
        print("PIPELINE COMPLETED")
        print("="*50)
        print(f"Submission saved to: submission.csv")
        print(f"Submission shape: {submission.shape}")
        print(f"Prediction distribution:")
        print(f"  Mean: {test_preds.mean():.6f}")
        print(f"  Std: {test_preds.std():.6f}")
        print(f"  Min: {test_preds.min():.6f}")
        print(f"  Max: {test_preds.max():.6f}")
        
        return submission

if __name__ == "__main__":
    # Initialize and run the predictor
    predictor = CreditDefaultPredictor()
    submission = predictor.run_full_pipeline()