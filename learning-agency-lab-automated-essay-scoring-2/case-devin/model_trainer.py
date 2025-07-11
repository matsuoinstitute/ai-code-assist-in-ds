import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

class EssayScorePredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.ensemble_weights = None
        self.feature_names = None
        
    def initialize_models(self):
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'ridge': Ridge(alpha=1.0)
        }
        
    def tune_hyperparameters(self, X_train, y_train, cv=3):
        print("Tuning hyperparameters...")
        
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5]
        }
        
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15]
        }
        
        ridge_params = {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        }
        
        param_grids = {
            'random_forest': rf_params,
            'xgboost': xgb_params,
            'ridge': ridge_params
        }
        
        best_models = {}
        
        for model_name, model in self.models.items():
            print(f"Tuning {model_name}...")
            
            grid_search = GridSearchCV(
                model, 
                param_grids[model_name],
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            
            if model_name == 'ridge':
                X_scaled = self.scaler.fit_transform(X_train)
                grid_search.fit(X_scaled, y_train)
            else:
                grid_search.fit(X_train, y_train)
                
            best_models[model_name] = grid_search.best_estimator_
            print(f"Best params for {model_name}: {grid_search.best_params_}")
            print(f"Best CV score for {model_name}: {-grid_search.best_score_:.4f}")
            
        self.models = best_models
        
    def train_models(self, X_train, y_train, tune_params=True):
        self.initialize_models()
        self.feature_names = X_train.columns.tolist()
        
        if tune_params:
            self.tune_hyperparameters(X_train, y_train)
        
        print("Training individual models...")
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            if model_name == 'ridge':
                X_scaled = self.scaler.fit_transform(X_train)
                model.fit(X_scaled, y_train)
            else:
                model.fit(X_train, y_train)
                
        print("Model training completed")
        
    def cross_validate_models(self, X_train, y_train, cv=5):
        print("Performing cross-validation...")
        
        cv_scores = {}
        
        for model_name, model in self.models.items():
            print(f"Cross-validating {model_name}...")
            
            if model_name == 'ridge':
                X_scaled = self.scaler.transform(X_train)
                scores = cross_val_score(
                    model, X_scaled, y_train, 
                    cv=cv, scoring='neg_mean_squared_error'
                )
            else:
                scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=cv, scoring='neg_mean_squared_error'
                )
                
            cv_scores[model_name] = {
                'mean_rmse': np.sqrt(-scores.mean()),
                'std_rmse': np.sqrt(scores.std()),
                'scores': scores
            }
            
            print(f"{model_name} CV RMSE: {cv_scores[model_name]['mean_rmse']:.4f} (+/- {cv_scores[model_name]['std_rmse']:.4f})")
            
        return cv_scores
    
    def calculate_ensemble_weights(self, X_val, y_val):
        print("Calculating ensemble weights...")
        
        predictions = {}
        errors = {}
        
        for model_name, model in self.models.items():
            if model_name == 'ridge':
                X_scaled = self.scaler.transform(X_val)
                pred = model.predict(X_scaled)
            else:
                pred = model.predict(X_val)
                
            predictions[model_name] = pred
            errors[model_name] = mean_squared_error(y_val, pred)
            
        inverse_errors = {name: 1.0 / error for name, error in errors.items()}
        total_inverse_error = sum(inverse_errors.values())
        
        self.ensemble_weights = {
            name: inv_error / total_inverse_error 
            for name, inv_error in inverse_errors.items()
        }
        
        print("Ensemble weights:")
        for name, weight in self.ensemble_weights.items():
            print(f"  {name}: {weight:.4f}")
            
        return self.ensemble_weights
    
    def predict(self, X_test):
        if not self.models:
            raise ValueError("Models not trained. Call train_models first.")
            
        predictions = {}
        
        for model_name, model in self.models.items():
            if model_name == 'ridge':
                X_scaled = self.scaler.transform(X_test)
                pred = model.predict(X_scaled)
            else:
                pred = model.predict(X_test)
                
            predictions[model_name] = pred
            
        if self.ensemble_weights is None:
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
        else:
            ensemble_pred = np.zeros(len(X_test))
            for model_name, pred in predictions.items():
                ensemble_pred += self.ensemble_weights[model_name] * pred
                
        ensemble_pred = np.clip(np.round(ensemble_pred), 1, 6).astype(int)
        
        return ensemble_pred, predictions
    
    def evaluate_models(self, X_val, y_val):
        print("Evaluating models on validation set...")
        
        ensemble_pred, individual_preds = self.predict(X_val)
        
        print(f"Ensemble RMSE: {np.sqrt(mean_squared_error(y_val, ensemble_pred)):.4f}")
        print(f"Ensemble MAE: {mean_absolute_error(y_val, ensemble_pred):.4f}")
        
        for model_name, pred in individual_preds.items():
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            mae = mean_absolute_error(y_val, pred)
            print(f"{model_name} RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            
        return ensemble_pred
    
    def save_models(self, filepath_prefix='essay_scorer'):
        for model_name, model in self.models.items():
            joblib.dump(model, f"{filepath_prefix}_{model_name}.pkl")
            
        joblib.dump(self.scaler, f"{filepath_prefix}_scaler.pkl")
        joblib.dump(self.ensemble_weights, f"{filepath_prefix}_weights.pkl")
        
        print(f"Models saved with prefix: {filepath_prefix}")
        
    def load_models(self, filepath_prefix='essay_scorer'):
        model_names = ['random_forest', 'xgboost', 'ridge']
        
        self.models = {}
        for model_name in model_names:
            self.models[model_name] = joblib.load(f"{filepath_prefix}_{model_name}.pkl")
            
        self.scaler = joblib.load(f"{filepath_prefix}_scaler.pkl")
        self.ensemble_weights = joblib.load(f"{filepath_prefix}_weights.pkl")
        
        print(f"Models loaded with prefix: {filepath_prefix}")
