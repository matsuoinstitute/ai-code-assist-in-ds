import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import re
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

def extract_features(text):
    """Extract features from text"""
    features = {}
    words = text.split()
    sentences = re.split(r'[.!?]+', text)

    features['word_count'] = len(words)
    features['char_count'] = len(text)
    features['sentence_count'] = len([s for s in sentences if s.strip()])
    features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
    features['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0
    features['comma_count'] = text.count(',')
    features['period_count'] = text.count('.')
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    unique_words = set(word.lower() for word in words)
    features['vocab_diversity'] = len(unique_words) / len(words) if words else 0
    features['caps_errors'] = len(re.findall(r'\b[a-z]+[A-Z]+\w*\b', text))
    features['repeated_chars'] = len(re.findall(r'(.)\1{2,}', text))

    return features

def analyze_data():
    """Perform comprehensive data analysis"""
    print("=== Essay Scoring Model - Validation Report ===\n")

    # Load data
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print("## Data Overview")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Score range: {train_df['score'].min()} - {train_df['score'].max()}")

    # Score distribution
    print("\n## Score Distribution")
    score_dist = train_df['score'].value_counts().sort_index()
    for score, count in score_dist.items():
        percentage = (count / len(train_df)) * 100
        print(f"Score {score}: {count:,} essays ({percentage:.1f}%)")

    # Text statistics
    print("\n## Text Statistics")
    train_df['word_count'] = train_df['full_text'].apply(lambda x: len(x.split()))
    train_df['char_count'] = train_df['full_text'].apply(len)

    print(f"Average word count: {train_df['word_count'].mean():.1f}")
    print(f"Average character count: {train_df['char_count'].mean():.1f}")
    print(f"Word count std: {train_df['word_count'].std():.1f}")

    # Extract features for analysis
    print("\nExtracting features for analysis...")
    train_features = [extract_features(text) for text in train_df['full_text']]
    train_feat_df = pd.DataFrame(train_features).fillna(0)

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words='english')
    tfidf_train = tfidf.fit_transform(train_df['full_text'])

    # Combine features
    scaler = StandardScaler()
    scaled_train_feat = scaler.fit_transform(train_feat_df)
    X_train = hstack([scaled_train_feat, tfidf_train])
    y_train = train_df['score'].values

    # Model comparison
    print("\n## Model Performance Comparison")
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Ridge Regression': Ridge(alpha=1.0)
    }

    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-cv_scores)
        results[name] = {
            'mean_rmse': rmse_scores.mean(),
            'std_rmse': rmse_scores.std(),
            'scores': rmse_scores
        }
        print(f"{name}:")
        print(f"  Cross-validation RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")

    # Best model analysis
    best_model_name = min(results, key=lambda x: results[x]['mean_rmse'])
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)

    print(f"\nBest model: {best_model_name}")

    # Feature importance (for Random Forest)
    if best_model_name == 'Random Forest':
        print("\n## Feature Importance (Top 10)")
        feature_names = list(train_feat_df.columns) + [f'tfidf_{i}' for i in range(500)]
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        for i in range(min(10, len(feature_names))):
            idx = indices[i]
            if idx < len(train_feat_df.columns):
                feature_name = feature_names[idx]
            else:
                feature_name = f"TF-IDF feature {idx - len(train_feat_df.columns)}"
            print(f"{i+1}. {feature_name}: {importances[idx]:.4f}")

    # Correlation analysis
    print("\n## Feature Correlation with Scores")
    correlations = {}
    for col in train_feat_df.columns:
        corr = np.corrcoef(train_feat_df[col], y_train)[0, 1]
        if not np.isnan(corr):
            correlations[col] = abs(corr)

    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    for feature, corr in sorted_corr[:10]:
        print(f"{feature}: {corr:.4f}")

    # Predictions analysis
    print("\n## Test Predictions Analysis")
    test_features = [extract_features(text) for text in test_df['full_text']]
    test_feat_df = pd.DataFrame(test_features).fillna(0)
    tfidf_test = tfidf.transform(test_df['full_text'])
    scaled_test_feat = scaler.transform(test_feat_df)
    X_test = hstack([scaled_test_feat, tfidf_test])

    predictions = best_model.predict(X_test)
    predictions_rounded = np.round(predictions).astype(int)
    predictions_clipped = np.clip(predictions_rounded, 1, 6)

    print("Raw predictions:", predictions)
    print("Rounded predictions:", predictions_rounded)
    print("Final predictions (clipped):", predictions_clipped)

    for i, essay_id in enumerate(test_df['essay_id']):
        print(f"Essay {essay_id}: Score {predictions_clipped[i]}")
        print(f"  Word count: {test_feat_df.iloc[i]['word_count']}")
        print(f"  Vocab diversity: {test_feat_df.iloc[i]['vocab_diversity']:.3f}")

    return results, best_model_name, train_feat_df, correlations

if __name__ == "__main__":
    results, best_model, features_df, correlations = analyze_data()
