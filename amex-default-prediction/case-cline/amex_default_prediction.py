"""
American Express Default Prediction
信用不履行予測モデル
"""

import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class AmexDefaultPredictor:
    def __init__(self):
        self.models = []
        self.feature_importance = None

    def load_data(self, sample_size=None):
        """データの読み込み"""
        print("Loading data...")

        # Sample submissionを確認してデータ形式を理解
        sample_sub = pd.read_csv('sample_submission.csv')
        print(f"Sample submission shape: {sample_sub.shape}")
        print(f"Sample submission columns: {list(sample_sub.columns)}")

        # ラベルデータの読み込み
        train_labels = pd.read_csv('train_labels.csv')
        print(f"Train labels shape: {train_labels.shape}")
        print(f"Train labels columns: {list(train_labels.columns)}")

        if sample_size:
            # サンプルサイズが指定された場合
            train_data = pd.read_csv('train_data.csv', nrows=sample_size)
            test_data = pd.read_csv('test_data.csv', nrows=sample_size//2)
        else:
            # 全データを読み込み（メモリに注意）
            print("Loading full training data...")
            train_data = pd.read_csv('train_data.csv')
            print("Loading test data...")
            test_data = pd.read_csv('test_data.csv')

        print(f"Train data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")

        return train_data, test_data, train_labels, sample_sub

    def feature_engineering(self, df):
        """特徴量エンジニアリング"""
        print("Feature engineering...")

        # 基本統計量の計算
        customer_features = []

        # 顧客IDでグループ化
        for customer_id, group in df.groupby('customer_ID'):
            features = {'customer_ID': customer_id}

            # 数値列の統計量
            numeric_cols = group.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'customer_ID']

            for col in numeric_cols:
                if col in group.columns:
                    values = group[col].dropna()
                    if len(values) > 0:
                        features[f'{col}_mean'] = values.mean()
                        features[f'{col}_std'] = values.std()
                        features[f'{col}_min'] = values.min()
                        features[f'{col}_max'] = values.max()
                        features[f'{col}_last'] = values.iloc[-1]
                        features[f'{col}_first'] = values.iloc[0]
                        if len(values) > 1:
                            features[f'{col}_diff'] = values.iloc[-1] - values.iloc[0]

            # カテゴリカル列の処理
            categorical_cols = group.select_dtypes(include=['object']).columns
            categorical_cols = [col for col in categorical_cols if col not in ['customer_ID', 'S_2']]

            for col in categorical_cols:
                if col in group.columns:
                    # 最頻値
                    mode_val = group[col].mode()
                    if len(mode_val) > 0:
                        features[f'{col}_mode'] = mode_val.iloc[0]
                    # ユニーク数
                    features[f'{col}_nunique'] = group[col].nunique()

            # 時系列特徴量（S_2が日付列の場合）
            if 'S_2' in group.columns:
                features['num_statements'] = len(group)
                # 日付の差分など
                if len(group) > 1:
                    try:
                        dates = pd.to_datetime(group['S_2'])
                        features['days_span'] = (dates.max() - dates.min()).days
                    except:
                        pass

            customer_features.append(features)

        # DataFrameに変換
        feature_df = pd.DataFrame(customer_features)

        print(f"Feature engineering completed. Shape: {feature_df.shape}")
        return feature_df

    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """LightGBMモデルの訓練"""
        print("Training LightGBM model...")

        # LightGBMパラメータ
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

        # データセットの準備
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = None
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # モデル訓練
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data] if valid_data else None,
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
        )

        return model

    def cross_validation(self, X, y, n_splits=5):
        """交差検証"""
        print(f"Starting {n_splits}-fold cross validation...")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Fold {fold + 1}/{n_splits}")

            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]

            # モデル訓練
            model = self.train_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold)

            # 予測
            val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)

            # スコア計算
            score = roc_auc_score(y_val_fold, val_pred)
            cv_scores.append(score)

            print(f"Fold {fold + 1} AUC: {score:.6f}")

            # モデル保存
            self.models.append(model)

            # メモリ解放
            del X_train_fold, X_val_fold, y_train_fold, y_val_fold
            gc.collect()

        print(f"CV Mean AUC: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores) * 2:.6f})")
        return cv_scores

    def predict(self, X_test):
        """テストデータの予測"""
        print("Making predictions...")

        predictions = np.zeros(len(X_test))

        for i, model in enumerate(self.models):
            pred = model.predict(X_test, num_iteration=model.best_iteration)
            predictions += pred
            print(f"Model {i+1}/{len(self.models)} prediction completed")

        # アンサンブル（平均）
        predictions /= len(self.models)

        return predictions

    def create_submission(self, test_predictions, sample_submission, filename='submission.csv'):
        """提出ファイルの作成"""
        submission = sample_submission.copy()
        submission['prediction'] = test_predictions
        submission.to_csv(filename, index=False)
        print(f"Submission file saved: {filename}")
        return submission

def main():
    """メイン実行関数"""
    print("=== American Express Default Prediction ===")

    # 予測器の初期化
    predictor = AmexDefaultPredictor()

    # データ読み込み（最初はサンプルで試す）
    print("Loading sample data for initial testing...")
    train_data, test_data, train_labels, sample_sub = predictor.load_data(sample_size=10000)

    # データの基本情報を表示
    print(f"\nData Info:")
    print(f"Train data columns: {len(train_data.columns)}")
    print(f"Unique customers in train: {train_data['customer_ID'].nunique()}")
    print(f"Target distribution: {train_labels['target'].value_counts()}")
    print(f"Target rate: {train_labels['target'].mean():.4f}")

    # 特徴量エンジニアリング
    print("\nProcessing training data...")
    train_features = predictor.feature_engineering(train_data)

    print("Processing test data...")
    test_features = predictor.feature_engineering(test_data)

    # ラベルとの結合
    train_features = train_features.merge(train_labels, on='customer_ID', how='left')

    # 特徴量とターゲットの分離
    feature_cols = [col for col in train_features.columns if col not in ['customer_ID', 'target']]
    X = train_features[feature_cols].fillna(0)
    y = train_features['target']

    print(f"Final feature shape: {X.shape}")
    print(f"Features: {len(feature_cols)}")

    # 交差検証
    cv_scores = predictor.cross_validation(X, y, n_splits=3)  # サンプルなので3分割

    # テストデータの予測
    X_test = test_features[feature_cols].fillna(0)
    test_predictions = predictor.predict(X_test)

    # 提出ファイル作成
    # テストデータのcustomer_IDを使用
    submission_df = pd.DataFrame({
        'customer_ID': test_features['customer_ID'],
        'prediction': test_predictions
    })

    # sample_submissionの形式に合わせる
    final_submission = sample_sub.merge(submission_df, on='customer_ID', how='left')
    final_submission['prediction'] = final_submission['prediction_y'].fillna(final_submission['prediction_x'])
    final_submission = final_submission[['customer_ID', 'prediction']]

    final_submission.to_csv('amex_submission.csv', index=False)
    print("Submission file created: amex_submission.csv")

    print("\n=== Model Training Completed ===")
    print(f"Cross-validation AUC: {np.mean(cv_scores):.6f}")

if __name__ == "__main__":
    main()
