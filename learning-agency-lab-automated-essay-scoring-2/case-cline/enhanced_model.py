import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import re

def extract_essay_features(text):
    """Extract comprehensive features from essay text"""
    features = {}

    # Basic length features
    features['char_count'] = len(text)
    features['word_count'] = len(text.split())
    features['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
    features['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])

    # Average lengths
    if features['sentence_count'] > 0:
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
    else:
        features['avg_sentence_length'] = 0

    # Punctuation and grammar features
    features['comma_count'] = text.count(',')
    features['semicolon_count'] = text.count(';')
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['quote_count'] = text.count('"') + text.count("'")

    # Vocabulary complexity
    words = re.findall(r'\b\w+\b', text.lower())
    if words:
        unique_words = set(words)
        features['unique_word_ratio'] = len(unique_words) / len(words)
        features['avg_word_length'] = np.mean([len(word) for word in words])

        # Long words (complexity indicator)
        long_words = [word for word in words if len(word) > 6]
        features['long_word_ratio'] = len(long_words) / len(words)
    else:
        features['unique_word_ratio'] = 0
        features['avg_word_length'] = 0
        features['long_word_ratio'] = 0

    # Text quality indicators
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(1, len(text))
    features['digit_count'] = sum(1 for c in text if c.isdigit())

    return features

print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Extract handcrafted features
print("Extracting essay features...")
train_features = []
for text in train_df['full_text']:
    train_features.append(extract_essay_features(text))

test_features = []
for text in test_df['full_text']:
    test_features.append(extract_essay_features(text))

train_feat_df = pd.DataFrame(train_features)
test_feat_df = pd.DataFrame(test_features)

# TF-IDF features
print("Creating TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    stop_words='english'
)

train_tfidf = tfidf.fit_transform(train_df['full_text'])
test_tfidf = tfidf.transform(test_df['full_text'])

# Convert TF-IDF to DataFrame
train_tfidf_df = pd.DataFrame(train_tfidf.toarray(),
                             columns=[f'tfidf_{i}' for i in range(train_tfidf.shape[1])])
test_tfidf_df = pd.DataFrame(test_tfidf.toarray(),
                            columns=[f'tfidf_{i}' for i in range(test_tfidf.shape[1])])

# Combine all features
X_train = pd.concat([train_feat_df, train_tfidf_df], axis=1)
X_test = pd.concat([test_feat_df, test_tfidf_df], axis=1)
y_train = train_df['score']

print(f"Combined feature matrix shape: {X_train.shape}")

# Train multiple models and select the best
models = {
    'Ridge': Ridge(alpha=1.0),
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42)
}

best_model = None
best_score = float('inf')
best_name = ""

print("Training and evaluating models...")
for name, model in models.items():
    print(f"Training {name}...")

    # Use scaled features for linear models
    if name == 'Ridge':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

    rmse = np.sqrt(-cv_scores.mean())
    print(f"{name} - Cross-validation RMSE: {rmse:.4f} (+/- {np.sqrt(cv_scores.var()):.4f})")

    if rmse < best_score:
        best_score = rmse
        best_model = model
        best_name = name

print(f"\nBest model: {best_name} with RMSE: {best_score:.4f}")

# Train the best model on full data
if best_name == 'Ridge':
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    best_model.fit(X_train_scaled, y_train)
    predictions = best_model.predict(X_test_scaled)
else:
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

# Ensure predictions are in valid range
predictions = np.clip(predictions, 1, 6)
predictions = np.round(predictions).astype(int)

print(f"Final predictions: {predictions}")

# Create submission
submission = pd.DataFrame({
    'essay_id': test_df['essay_id'],
    'score': predictions
})

submission.to_csv('enhanced_submission.csv', index=False)
print("Enhanced submission saved to enhanced_submission.csv")
print("\nSubmission content:")
print(submission)
