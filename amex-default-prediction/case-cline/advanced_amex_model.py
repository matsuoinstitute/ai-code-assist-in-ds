"""
Advanced American Express Default Prediction Model
高度な特徴量エンジニアリングとアンサンブルモデル
"""

import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

class AdvancedAmexPredictor:
    def __init__(self):
        self.models = []
        self.feature_importance = {}
        self.label_encoders = {}

    def load_data(self, sample_size=None):
        """データの読み込み"""
        print("Loading data...")

        # ラベルデータの読み込み
        train_labels = pd.read_csv('train_labels.csv')
        sample_sub = pd.read_csv('sample_submission.csv')

        if sample_size:
            print(f"Loading sample data (size: {sample_size})...")
            train_data = pd.read_csv('train_data.csv', nrows=sample_size)
            test_data = pd.read_csv('test_data.csv', nrows=sample_size//2)
        else:
            print("Loading full data...")
            train_data = pd.read_csv('train_data.csv')
            test_data = pd.read_csv('test_data.csv')

        print(f"Train data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        print(f"Train labels shape: {train_labels.shape}")
        print(f"Target rate: {train_labels['target'].mean():.4f}")

        return train_data, test_data, train_labels, sample_sub

    def advanced_feature_engineering(self, df, is_train=True):
        """高度な特徴量エンジニアリング"""
        print("Advanced feature engineering...")

        # カテゴリカル変数のエンコーディング
        categorical_cols = ['D_63', 'D_64']
        for col in categorical_cols:
            if col in df.columns:
                if is_train:
                    le = LabelEncoder()
                    df[col] = df[col].fillna('missing')
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        df[col] = df[col].fillna('missing')
                        # 未知のカテゴリは-1に設定
                        df[col] = df[col].astype(str)
                        known_classes = set(self.label_encoders[col].classes_)
                        df[col] = df[col].apply(lambda x: x if x in known_classes else 'unknown')

                        # unknownクラスを追加
                        if 'unknown' not in self.label_encoders[col].classes_:
                            self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, 'unknown')

                        df[col] = self.label_encoders[col].transform(df[col])

        # 日付特徴量の処理
        if 'S_2' in df.columns:
            df['S_2'] = pd.to_datetime(df['S_2'])
            df['S_2_year'] = df['S_2'].dt.year
            df['S_2_month'] = df['S_2'].dt.month
            df['S_2_day'] = df['S_2'].dt.day
            df['S_2_dayofweek'] = df['S_2'].dt.dayofweek
            df['S_2_quarter'] = df['S_2'].dt.quarter

        # 顧客レベルの特徴量集約
        customer_features = []

        print("Processing customer-level features...")
        for customer_id, group in df.groupby('customer_ID'):
            features = {'customer_ID': customer_id}

            # 基本統計量
            numeric_cols = group.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['customer_ID']]

            for col in numeric_cols:
                if col in group.columns:
                    values = group[col].dropna()
                    if len(values) > 0:
                        # 基本統計量
                        features[f'{col}_mean'] = values.mean()
                        features[f'{col}_std'] = values.std()
                        features[f'{col}_min'] = values.min()
                        features[f'{col}_max'] = values.max()
                        features[f'{col}_median'] = values.median()
                        features[f'{col}_last'] = values.iloc[-1]
                        features[f'{col}_first'] = values.iloc[0]

                        # 変化量
                        if len(values) > 1:
                            features[f'{col}_diff'] = values.iloc[-1] - values.iloc[0]
                            features[f'{col}_slope'] = np.polyfit(range(len(values)), values, 1)[0]

                        # パーセンタイル
                        features[f'{col}_q25'] = values.quantile(0.25)
                        features[f'{col}_q75'] = values.quantile(0.75)
                        features[f'{col}_iqr'] = features[f'{col}_q75'] - features[f'{col}_q25']

                        # 欠損値の割合
                        features[f'{col}_missing_ratio'] = group[col].isnull().mean()

                        # 変動係数
                        if features[f'{col}_mean'] != 0:
                            features[f'{col}_cv'] = features[f'{col}_std'] / abs(features[f'{col}_mean'])

            # 時系列特徴量
            if 'S_2' in group.columns:
                features['num_statements'] = len(group)
                if len(group) > 1:
                    dates = pd.to_datetime(group['S_2'])
                    features['days_span'] = (dates.max() - dates.min()).days
                    features['avg_days_between_statements'] = features['days_span'] / (len(group) - 1)

                # 最新の月、四半期
                if 'S_2_month' in group.columns:
                    features['last_month'] = group['S_2_month'].iloc[-1]
                if 'S_2_quarter' in group.columns:
                    features['last_quarter'] = group['S_2_quarter'].iloc[-1]

            # カテゴリカル特徴量
            for col in ['D_63', 'D_64']:
                if col in group.columns:
                    features[f'{col}_mode'] = group[col].mode().iloc[0] if len(group[col].mode()) > 0 else -1
                    features[f'{col}_nunique'] = group[col].nunique()

            customer_features.append(features)

        # DataFrameに変換
        feature_df = pd.DataFrame(customer_features)

        # 無限大値とNaNの処理
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

        print(f"Feature engineering completed. Shape: {feature_df.shape}")
        return feature_df

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """LightGBMモデルの訓練"""
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
            'verbose': -1,
            'random_state': 42
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )

        return model

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """XGBoostモデルの訓練"""
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 0
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, 'eval')],
            early_stopping_rounds=100,
            verbose_eval=0
        )

        return model

    def train_catboost(self, X_train, y_train, X_val, y_val):
        """CatBoostモデルの訓練"""
        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,
            depth=6,
            eval_metric='AUC',
            random_seed=42,
            verbose=0,
            early_stopping_rounds=100
        )

        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )

        return model

    def cross_validation_ensemble(self, X, y, n_splits=5):
        """アンサンブルモデルの交差検証"""
        print(f"Starting {n_splits}-fold cross validation with ensemble...")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []

        lgb_models = []
        xgb_models = []
        cat_models = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Fold {fold + 1}/{n_splits}")

            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]

            # LightGBM
            print("  Training LightGBM...")
            lgb_model = self.train_lightgbm(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            lgb_pred = lgb_model.predict(X_val_fold, num_iteration=lgb_model.best_iteration)
            lgb_models.append(lgb_model)

            # XGBoost
            print("  Training XGBoost...")
            xgb_model = self.train_xgboost(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            xgb_pred = xgb_model.predict(xgb.DMatrix(X_val_fold))
            xgb_models.append(xgb_model)

            # CatBoost
            print("  Training CatBoost...")
            cat_model = self.train_catboost(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            cat_pred = cat_model.predict_proba(X_val_fold)[:, 1]
            cat_models.append(cat_model)

            # アンサンブル予測
            ensemble_pred = (lgb_pred + xgb_pred + cat_pred) / 3

            # スコア計算
            score = roc_auc_score(y_val_fold, ensemble_pred)
            cv_scores.append(score)

            print(f"  Fold {fold + 1} AUC: {score:.6f}")

            # メモリ解放
            del X_train_fold, X_val_fold, y_train_fold, y_val_fold
            gc.collect()

        self.models = {
            'lightgbm': lgb_models,
            'xgboost': xgb_models,
            'catboost': cat_models
        }

        print(f"CV Mean AUC: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores) * 2:.6f})")
        return cv_scores

    def predict_ensemble(self, X_test):
        """アンサンブル予測"""
        print("Making ensemble predictions...")

        lgb_preds = np.zeros(len(X_test))
        xgb_preds = np.zeros(len(X_test))
        cat_preds = np.zeros(len(X_test))

        # LightGBM予測
        for i, model in enumerate(self.models['lightgbm']):
            pred = model.predict(X_test, num_iteration=model.best_iteration)
            lgb_preds += pred
        lgb_preds /= len(self.models['lightgbm'])

        # XGBoost予測
        for i, model in enumerate(self.models['xgboost']):
            pred = model.predict(xgb.DMatrix(X_test))
            xgb_preds += pred
        xgb_preds /= len(self.models['xgboost'])

        # CatBoost予測
        for i, model in enumerate(self.models['catboost']):
            pred = model.predict_proba(X_test)[:, 1]
            cat_preds += pred
        cat_preds /= len(self.models['catboost'])

        # 最終アンサンブル
        final_pred = (lgb_preds + xgb_preds + cat_preds) / 3

        return final_pred

def main():
    """メイン実行関数"""
    print("=== Advanced American Express Default Prediction ===")

    # 予測器の初期化
    predictor = AdvancedAmexPredictor()

    # データ読み込み（サンプルサイズを調整）
    print("Loading data...")
    train_data, test_data, train_labels, sample_sub = predictor.load_data(sample_size=50000)

    # 高度な特徴量エンジニアリング
    print("\nProcessing training data...")
    train_features = predictor.advanced_feature_engineering(train_data, is_train=True)

    print("Processing test data...")
    test_features = predictor.advanced_feature_engineering(test_data, is_train=False)

    # ラベルとの結合
    train_features = train_features.merge(train_labels, on='customer_ID', how='left')

    # 特徴量とターゲットの分離
    feature_cols = [col for col in train_features.columns if col not in ['customer_ID', 'target']]
    X = train_features[feature_cols].fillna(0)
    y = train_features['target']

    print(f"Final feature shape: {X.shape}")
    print(f"Number of features: {len(feature_cols)}")

    # アンサンブル交差検証
    cv_scores = predictor.cross_validation_ensemble(X, y, n_splits=3)

    # テストデータの予測
    X_test = test_features[feature_cols].fillna(0)
    test_predictions = predictor.predict_ensemble(X_test)

    # 提出ファイル作成
    submission_df = pd.DataFrame({
        'customer_ID': test_features['customer_ID'],
        'prediction': test_predictions
    })

    # sample_submissionの形式に合わせる
    final_submission = sample_sub.merge(submission_df, on='customer_ID', how='left')
    final_submission['prediction'] = final_submission['prediction_y'].fillna(0.5)
    final_submission = final_submission[['customer_ID', 'prediction']]

    final_submission.to_csv('advanced_amex_submission.csv', index=False)
    print("Advanced submission file created: advanced_amex_submission.csv")

    print("\n=== Advanced Model Training Completed ===")
    print(f"Cross-validation AUC: {np.mean(cv_scores):.6f}")
    print(f"Submission shape: {final_submission.shape}")
    print(f"Prediction range: [{final_submission['prediction'].min():.4f}, {final_submission['prediction'].max():.4f}]")

if __name__ == "__main__":
    main()
