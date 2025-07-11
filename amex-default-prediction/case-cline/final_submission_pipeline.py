"""
Final American Express Default Prediction Pipeline
最終的な提出ファイル作成パイプライン
"""

import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class FinalAmexPredictor:
    def __init__(self):
        self.models = []
        self.label_encoders = {}
        self.results = {}

    def log(self, message):
        """ログメッセージを記録"""
        print(message)
        if 'log' not in self.results:
            self.results['log'] = []
        self.results['log'].append(message)

    def load_data(self, sample_size=100000):
        """データの読み込み"""
        self.log("=== Loading Data ===")

        # ラベルデータの読み込み
        train_labels = pd.read_csv('train_labels.csv')
        sample_sub = pd.read_csv('sample_submission.csv')

        self.log(f"Train labels shape: {train_labels.shape}")
        self.log(f"Sample submission shape: {sample_sub.shape}")
        self.log(f"Target rate: {train_labels['target'].mean():.4f}")

        # サンプルデータの読み込み
        self.log(f"Loading sample data (size: {sample_size})...")
        train_data = pd.read_csv('train_data.csv', nrows=sample_size)
        test_data = pd.read_csv('test_data.csv', nrows=sample_size//2)

        self.log(f"Train data shape: {train_data.shape}")
        self.log(f"Test data shape: {test_data.shape}")

        return train_data, test_data, train_labels, sample_sub

    def feature_engineering(self, df, is_train=True):
        """特徴量エンジニアリング"""
        self.log("=== Feature Engineering ===")

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
                        df[col] = df[col].astype(str)
                        known_classes = set(self.label_encoders[col].classes_)
                        df[col] = df[col].apply(lambda x: x if x in known_classes else 'missing')
                        df[col] = self.label_encoders[col].transform(df[col])

        # 日付特徴量の処理
        if 'S_2' in df.columns:
            df['S_2'] = pd.to_datetime(df['S_2'])
            df['S_2_year'] = df['S_2'].dt.year
            df['S_2_month'] = df['S_2'].dt.month
            df['S_2_dayofweek'] = df['S_2'].dt.dayofweek

        # 顧客レベルの特徴量集約
        customer_features = []

        self.log("Processing customer-level features...")
        for customer_id, group in df.groupby('customer_ID'):
            features = {'customer_ID': customer_id}

            # 数値列の統計量
            numeric_cols = group.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['customer_ID']]

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
                        features[f'{col}_missing_ratio'] = group[col].isnull().mean()

            # 時系列特徴量
            if 'S_2' in group.columns:
                features['num_statements'] = len(group)
                if len(group) > 1:
                    dates = pd.to_datetime(group['S_2'])
                    features['days_span'] = (dates.max() - dates.min()).days

            # カテゴリカル特徴量
            for col in ['D_63', 'D_64']:
                if col in group.columns:
                    mode_val = group[col].mode()
                    features[f'{col}_mode'] = mode_val.iloc[0] if len(mode_val) > 0 else -1
                    features[f'{col}_nunique'] = group[col].nunique()

            customer_features.append(features)

        # DataFrameに変換
        feature_df = pd.DataFrame(customer_features)
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

        self.log(f"Feature engineering completed. Shape: {feature_df.shape}")
        return feature_df

    def train_models(self, X, y):
        """複数モデルの訓練"""
        self.log("=== Training Models ===")

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            self.log(f"Fold {fold + 1}/3")

            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]

            # LightGBM
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

            train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
            valid_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )

            # 予測
            val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
            score = roc_auc_score(y_val_fold, val_pred)
            cv_scores.append(score)

            self.log(f"  Fold {fold + 1} AUC: {score:.6f}")
            self.models.append(model)

            # メモリ解放
            del X_train_fold, X_val_fold, y_train_fold, y_val_fold
            gc.collect()

        mean_score = np.mean(cv_scores)
        self.log(f"CV Mean AUC: {mean_score:.6f} (+/- {np.std(cv_scores) * 2:.6f})")
        self.results['cv_auc'] = mean_score

        return cv_scores

    def predict(self, X_test):
        """テストデータの予測"""
        self.log("=== Making Predictions ===")

        predictions = np.zeros(len(X_test))

        for i, model in enumerate(self.models):
            pred = model.predict(X_test, num_iteration=model.best_iteration)
            predictions += pred

        predictions /= len(self.models)

        self.log(f"Predictions completed. Range: [{predictions.min():.6f}, {predictions.max():.6f}]")
        return predictions

    def create_submission(self, test_features, predictions, sample_sub):
        """提出ファイルの作成"""
        self.log("=== Creating Submission ===")

        submission_df = pd.DataFrame({
            'customer_ID': test_features['customer_ID'],
            'prediction': predictions
        })

        # sample_submissionの形式に合わせる
        final_submission = sample_sub.merge(submission_df, on='customer_ID', how='left')
        final_submission['prediction'] = final_submission['prediction_y'].fillna(0.5)
        final_submission = final_submission[['customer_ID', 'prediction']]

        # 提出ファイル保存
        final_submission.to_csv('final_amex_submission.csv', index=False)

        self.log(f"Submission file created: final_amex_submission.csv")
        self.log(f"Submission shape: {final_submission.shape}")
        self.log(f"Prediction statistics:")
        self.log(f"  Mean: {final_submission['prediction'].mean():.6f}")
        self.log(f"  Std: {final_submission['prediction'].std():.6f}")
        self.log(f"  Min: {final_submission['prediction'].min():.6f}")
        self.log(f"  Max: {final_submission['prediction'].max():.6f}")

        return final_submission

    def save_report(self):
        """結果レポートの保存"""
        with open('final_report.txt', 'w') as f:
            f.write("=== American Express Default Prediction - Final Report ===\n\n")

            if 'log' in self.results:
                for log_entry in self.results['log']:
                    f.write(log_entry + '\n')

            f.write(f"\n=== Summary ===\n")
            f.write(f"Cross-validation AUC: {self.results.get('cv_auc', 'N/A'):.6f}\n")
            f.write(f"Number of models: {len(self.models)}\n")
            f.write(f"Submission file: final_amex_submission.csv\n")

        self.log("Report saved: final_report.txt")

def main():
    """メイン実行関数"""
    predictor = FinalAmexPredictor()

    try:
        # データ読み込み
        train_data, test_data, train_labels, sample_sub = predictor.load_data(sample_size=50000)

        # 特徴量エンジニアリング
        train_features = predictor.feature_engineering(train_data, is_train=True)
        test_features = predictor.feature_engineering(test_data, is_train=False)

        # ラベルとの結合
        train_features = train_features.merge(train_labels, on='customer_ID', how='left')

        # 特徴量とターゲットの分離
        feature_cols = [col for col in train_features.columns if col not in ['customer_ID', 'target']]
        X = train_features[feature_cols].fillna(0)
        y = train_features['target']

        predictor.log(f"Final feature shape: {X.shape}")
        predictor.log(f"Number of features: {len(feature_cols)}")

        # モデル訓練
        cv_scores = predictor.train_models(X, y)

        # テストデータの予測
        X_test = test_features[feature_cols].fillna(0)
        test_predictions = predictor.predict(X_test)

        # 提出ファイル作成
        final_submission = predictor.create_submission(test_features, test_predictions, sample_sub)

        # レポート保存
        predictor.save_report()

        predictor.log("=== Pipeline Completed Successfully ===")

    except Exception as e:
        predictor.log(f"Error occurred: {e}")
        import traceback
        predictor.log(traceback.format_exc())
        predictor.save_report()

if __name__ == "__main__":
    main()
