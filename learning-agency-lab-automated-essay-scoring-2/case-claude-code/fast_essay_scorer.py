import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def extract_features(text):
    """Extract basic features from text"""
    features = {}

    # Basic length features
    words = text.split()
    sentences = re.split(r'[.!?]+', text)

    features['word_count'] = len(words)
    features['char_count'] = len(text)
    features['sentence_count'] = len([s for s in sentences if s.strip()])
    features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
    features['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0

    # Punctuation
    features['comma_count'] = text.count(',')
    features['period_count'] = text.count('.')
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')

    # Vocabulary diversity
    unique_words = set(word.lower() for word in words)
    features['vocab_diversity'] = len(unique_words) / len(words) if words else 0

    # Error indicators
    features['caps_errors'] = len(re.findall(r'\b[a-z]+[A-Z]+\w*\b', text))
    features['repeated_chars'] = len(re.findall(r'(.)\1{2,}', text))

    return features

def main():
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    print(f"Score distribution:\n{train_df['score'].value_counts().sort_index()}")

    print("\nExtracting features...")

    # Extract handcrafted features
    train_features = [extract_features(text) for text in train_df['full_text']]
    test_features = [extract_features(text) for text in test_df['full_text']]

    train_feat_df = pd.DataFrame(train_features).fillna(0)
    test_feat_df = pd.DataFrame(test_features).fillna(0)

    # TF-IDF features (reduced size for speed)
    print("Computing TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words='english')
    tfidf_train = tfidf.fit_transform(train_df['full_text'])
    tfidf_test = tfidf.transform(test_df['full_text'])

    # Combine features
    scaler = StandardScaler()
    scaled_train_feat = scaler.fit_transform(train_feat_df)
    scaled_test_feat = scaler.transform(test_feat_df)

    from scipy.sparse import hstack
    X_train = hstack([scaled_train_feat, tfidf_train])
    X_test = hstack([scaled_test_feat, tfidf_test])
    y_train = train_df['score'].values

    print(f"Feature matrix shape: {X_train.shape}")

    # Split for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print("\nTraining models...")

    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_split, y_train_split)
    rf_pred = rf_model.predict(X_val_split)
    rf_rmse = np.sqrt(mean_squared_error(y_val_split, rf_pred))

    # Train Ridge Regression
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_split, y_train_split)
    ridge_pred = ridge_model.predict(X_val_split)
    ridge_rmse = np.sqrt(mean_squared_error(y_val_split, ridge_pred))

    print(f"Random Forest RMSE: {rf_rmse:.4f}")
    print(f"Ridge Regression RMSE: {ridge_rmse:.4f}")

    # Choose best model
    if rf_rmse < ridge_rmse:
        best_model = rf_model
        best_name = "Random Forest"
    else:
        best_model = ridge_model
        best_name = "Ridge Regression"

    print(f"Best model: {best_name}")

    # Retrain on full training set
    print("Retraining on full dataset...")
    best_model.fit(X_train, y_train)

    # Generate predictions
    print("Generating predictions...")
    predictions = best_model.predict(X_test)
    predictions = np.round(predictions).astype(int)
    predictions = np.clip(predictions, 0, 4)

    # Save predictions
    submission = pd.DataFrame({
        'essay_id': test_df['essay_id'],
        'score': predictions
    })
    submission.to_csv('test_predictions.csv', index=False)

    print(f"Predictions saved to test_predictions.csv")
    print(f"Prediction distribution:\n{pd.Series(predictions).value_counts().sort_index()}")

if __name__ == "__main__":
    main()
