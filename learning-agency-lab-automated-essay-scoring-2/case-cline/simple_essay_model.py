import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import re

def extract_basic_features(text):
    """Extract basic features from essay text"""
    features = {}

    # Basic length features
    features['char_count'] = len(text)
    features['word_count'] = len(text.split())
    features['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
    features['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])

    # Average lengths
    features['avg_sentence_length'] = features['word_count'] / max(1, features['sentence_count'])

    # Punctuation features
    features['comma_count'] = text.count(',')
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')

    # Vocabulary complexity
    words = re.findall(r'\b\w+\b', text.lower())
    unique_words = set(words)
    features['unique_word_ratio'] = len(unique_words) / max(1, len(words))
    features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0

    return features

def preprocess_text(text):
    """Clean text for TF-IDF"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = ' '.join(text.split())
    return text

def main():
    with open('model_output.txt', 'w') as f:
        f.write("Starting essay scoring model...\n")

        # Load data
        f.write("Loading data...\n")
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')

        f.write(f"Train shape: {train_df.shape}\n")
        f.write(f"Test shape: {test_df.shape}\n")

        # Extract features
        f.write("Extracting features...\n")
        train_features = []
        for text in train_df['full_text']:
            train_features.append(extract_basic_features(text))

        test_features = []
        for text in test_df['full_text']:
            test_features.append(extract_basic_features(text))

        # Convert to DataFrames
        train_feat_df = pd.DataFrame(train_features)
        test_feat_df = pd.DataFrame(test_features)

        # TF-IDF
        f.write("Creating TF-IDF features...\n")
        train_texts = [preprocess_text(text) for text in train_df['full_text']]
        test_texts = [preprocess_text(text) for text in test_df['full_text']]

        tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
        train_tfidf = tfidf.fit_transform(train_texts)
        test_tfidf = tfidf.transform(test_texts)

        # Combine features
        train_tfidf_df = pd.DataFrame(train_tfidf.toarray(),
                                    columns=[f'tfidf_{i}' for i in range(train_tfidf.shape[1])])
        test_tfidf_df = pd.DataFrame(test_tfidf.toarray(),
                                   columns=[f'tfidf_{i}' for i in range(test_tfidf.shape[1])])

        X_train = pd.concat([train_feat_df, train_tfidf_df], axis=1)
        X_test = pd.concat([test_feat_df, test_tfidf_df], axis=1)
        y_train = train_df['score']

        f.write(f"Feature matrix shape: {X_train.shape}\n")

        # Train models
        f.write("Training models...\n")

        models = {
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
        }

        best_model = None
        best_score = float('inf')

        for name, model in models.items():
            f.write(f"Training {name}...\n")

            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-cv_scores)
            avg_rmse = np.mean(rmse_scores)

            f.write(f"{name} - Average RMSE: {avg_rmse:.4f}\n")

            if avg_rmse < best_score:
                best_score = avg_rmse
                best_model = model
                best_model_name = name

        f.write(f"\nBest model: {best_model_name} with RMSE: {best_score:.4f}\n")

        # Train best model and predict
        best_model.fit(X_train, y_train)
        predictions = best_model.predict(X_test)

        # Clip and round predictions
        predictions = np.clip(predictions, 1, 6)
        predictions = np.round(predictions).astype(int)

        f.write(f"Predictions: {predictions}\n")

        # Create submission
        submission = pd.DataFrame({
            'essay_id': test_df['essay_id'],
            'score': predictions
        })

        submission.to_csv('submission.csv', index=False)
        f.write("Submission file created: submission.csv\n")

        # Show submission content
        f.write("\nSubmission content:\n")
        f.write(str(submission))

if __name__ == "__main__":
    main()
