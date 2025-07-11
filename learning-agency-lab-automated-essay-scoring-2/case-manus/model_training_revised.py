import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
import lightgbm as lgb
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# NLTKのリソースをダウンロード
nltk.download('punkt')
nltk.download('stopwords')

# データの読み込み
print("データを読み込んでいます...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# テキスト前処理関数
def preprocess_text(text):
    # 小文字化
    text = text.lower()
    # 句読点の除去
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    # 数字の除去
    text = re.sub(r'\d+', '', text)
    # 余分な空白の除去
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 文の数を数える簡易関数（nltk.sent_tokenizeの代わり）
def count_sentences(text):
    # ピリオド、感嘆符、疑問符で文を分割
    sentences = re.split(r'[.!?]+', text)
    # 空の文字列を除外
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)

# 特徴量エンジニアリング関数
def extract_features(text):
    # 前処理済みテキスト
    processed_text = preprocess_text(text)
    
    # 基本的な統計量
    word_count = len(processed_text.split())
    sent_count = count_sentences(text)  # 独自の文カウント関数を使用
    char_count = len(text)
    avg_word_length = char_count / (word_count + 1)  # ゼロ除算防止
    avg_sent_length = word_count / (sent_count + 1)  # ゼロ除算防止
    
    # 語彙の多様性（ユニークな単語の割合）
    unique_words = set(processed_text.split())
    vocab_diversity = len(unique_words) / (word_count + 1)  # ゼロ除算防止
    
    # 文法的特徴（大文字で始まる文の割合）
    original_sentences = re.split(r'[.!?]+', text)
    original_sentences = [s.strip() for s in original_sentences if s.strip()]
    capital_sent_ratio = sum(1 for s in original_sentences if s and s[0].isupper()) / (sent_count + 1)
    
    # 句読点の使用率
    punct_count = sum(1 for char in text if char in string.punctuation)
    punct_ratio = punct_count / (char_count + 1)  # ゼロ除算防止
    
    return {
        'word_count': word_count,
        'sent_count': sent_count,
        'char_count': char_count,
        'avg_word_length': avg_word_length,
        'avg_sent_length': avg_sent_length,
        'vocab_diversity': vocab_diversity,
        'capital_sent_ratio': capital_sent_ratio,
        'punct_ratio': punct_ratio
    }

# 特徴量の抽出
print("特徴量を抽出しています...")
train_features = pd.DataFrame([extract_features(text) for text in train_df['full_text']])
test_features = pd.DataFrame([extract_features(text) for text in test_df['full_text']])

# TF-IDF特徴量の抽出
print("TF-IDF特徴量を抽出しています...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=5,
    max_df=0.8,
    preprocessor=preprocess_text
)
tfidf_train = tfidf_vectorizer.fit_transform(train_df['full_text'])
tfidf_test = tfidf_vectorizer.transform(test_df['full_text'])

# TF-IDF特徴量をデータフレームに変換
tfidf_train_df = pd.DataFrame(
    tfidf_train.toarray(),
    columns=[f'tfidf_{i}' for i in range(tfidf_train.shape[1])]
)
tfidf_test_df = pd.DataFrame(
    tfidf_test.toarray(),
    columns=[f'tfidf_{i}' for i in range(tfidf_test.shape[1])]
)

# 特徴量の結合
X_train = pd.concat([train_features, tfidf_train_df], axis=1)
X_test = pd.concat([test_features, tfidf_test_df], axis=1)
y_train = train_df['score']

# 訓練データとバリデーションデータの分割
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# モデルの定義
models = {
    'Ridge': Ridge(random_state=42),
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42),
    'LightGBM': lgb.LGBMRegressor(random_state=42)
}

# ハイパーパラメータグリッド
param_grids = {
    'Ridge': {
        'alpha': [0.1, 1.0, 10.0]
    },
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20]
    },
    'GradientBoosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    },
    'LightGBM': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }
}

# モデルの訓練と評価
results = {}
best_models = {}

print("モデルの訓練と評価を行っています...")
for name, model in models.items():
    print(f"{name}モデルのグリッドサーチを実行中...")
    grid_search = GridSearchCV(
        model,
        param_grids[name],
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train_split, y_train_split)
    
    # 最適モデルの保存
    best_model = grid_search.best_estimator_
    best_models[name] = best_model
    
    # バリデーションデータでの予測
    y_val_pred = best_model.predict(X_val)
    
    # 評価指標の計算
    mse = mean_squared_error(y_val, y_val_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_val_pred)
    
    # 結果の保存
    results[name] = {
        'best_params': grid_search.best_params_,
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }
    
    print(f"{name}の結果:")
    print(f"  最適パラメータ: {grid_search.best_params_}")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print()

# 最良モデルの選択
best_model_name = min(results, key=lambda x: results[x]['rmse'])
best_model = best_models[best_model_name]

print(f"最良モデル: {best_model_name}")
print(f"RMSE: {results[best_model_name]['rmse']:.4f}")

# 最良モデルの保存
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# テストデータに対する予測
y_test_pred = best_model.predict(X_test)

# 予測結果の保存
test_predictions = pd.DataFrame({
    'essay_id': test_df['essay_id'],
    'predicted_score': y_test_pred
})
test_predictions.to_csv('test_predictions.csv', index=False)

print("モデル訓練と予測が完了しました。")
print(f"テストデータの予測結果: {test_predictions}")
