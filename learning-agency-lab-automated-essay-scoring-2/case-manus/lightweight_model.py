import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

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

# 文の数を数える簡易関数
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
    sent_count = count_sentences(text)
    char_count = len(text)
    avg_word_length = char_count / (word_count + 1)  # ゼロ除算防止
    avg_sent_length = word_count / (sent_count + 1)  # ゼロ除算防止
    
    # 語彙の多様性（ユニークな単語の割合）
    unique_words = set(processed_text.split())
    vocab_diversity = len(unique_words) / (word_count + 1)  # ゼロ除算防止
    
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
        'punct_ratio': punct_ratio
    }

# 特徴量の抽出
print("特徴量を抽出しています...")
train_features = pd.DataFrame([extract_features(text) for text in train_df['full_text']])
test_features = pd.DataFrame([extract_features(text) for text in test_df['full_text']])

# TF-IDF特徴量の抽出（次元数を削減）
print("TF-IDF特徴量を抽出しています（次元数を削減）...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=500,  # 次元数を大幅に削減
    min_df=10,
    max_df=0.7,
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

# Ridgeモデルの訓練（グリッドサーチなし）
print("Ridgeモデルを訓練しています...")
ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train_split, y_train_split)

# バリデーションデータでの予測と評価
y_val_pred = ridge_model.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, y_val_pred)

print(f"Ridge モデルの評価結果:")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE: {mae:.4f}")

# モデルの保存
joblib.dump(ridge_model, 'ridge_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# テストデータに対する予測
y_test_pred = ridge_model.predict(X_test)

# 予測結果の保存
test_predictions = pd.DataFrame({
    'essay_id': test_df['essay_id'],
    'predicted_score': y_test_pred
})
test_predictions.to_csv('test_predictions.csv', index=False)

print("モデル訓練と予測が完了しました。")
print(f"テストデータの予測結果:\n{test_predictions}")
