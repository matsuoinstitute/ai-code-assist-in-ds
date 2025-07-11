import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    print("NLTK download failed, continuing without some features")

class EssayFeatureExtractor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()

    def extract_features(self, text):
        """Extract comprehensive features from essay text"""
        features = {}

        # Basic length features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())

        try:
            sentences = sent_tokenize(text)
            features['sentence_count'] = len(sentences)
            features['avg_sentence_length'] = features['word_count'] / max(1, features['sentence_count'])
        except:
            features['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
            features['avg_sentence_length'] = features['word_count'] / max(1, features['sentence_count'])

        # Paragraph count
        features['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])

        # Punctuation features
        features['comma_count'] = text.count(',')
        features['semicolon_count'] = text.count(';')
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['quote_count'] = text.count('"') + text.count("'")

        # Vocabulary complexity
        words = re.findall(r'\b\w+\b', text.lower())
        unique_words = set(words)
        features['unique_word_ratio'] = len(unique_words) / max(1, len(words))
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0

        # Long words (complexity indicator)
        long_words = [word for word in words if len(word) > 6]
        features['long_word_ratio'] = len(long_words) / max(1, len(words))

        # Spelling and grammar indicators (simple heuristics)
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(1, len(text))
        features['digit_count'] = sum(1 for c in text if c.isdigit())

        # Readability approximation
        if features['sentence_count'] > 0 and features['word_count'] > 0:
            # Simple Flesch-like score
            avg_sentence_len = features['word_count'] / features['sentence_count']
            avg_syllables = features['avg_word_length'] * 0.5  # rough approximation
            features['readability_score'] = 206.835 - (1.015 * avg_sentence_len) - (84.6 * avg_syllables)
        else:
            features['readability_score'] = 0

        return features

def preprocess_text(text):
    """Clean and preprocess text for TF-IDF"""
    # Convert to lowercase
    text = text.lower()

    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text

def train_essay_scoring_model():
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print("Extracting features...")
    feature_extractor = EssayFeatureExtractor()

    # Extract handcrafted features
    train_features = []
    for text in train_df['full_text']:
        train_features.append(feature_extractor.extract_features(text))

    test_features = []
    for text in test_df['full_text']:
        test_features.append(feature_extractor.extract_features(text))

    # Convert to DataFrames
    train_feat_df = pd.DataFrame(train_features)
    test_feat_df = pd.DataFrame(test_features)

    # Preprocess text for TF-IDF
    train_texts = [preprocess_text(text) for text in train_df['full_text']]
    test_texts = [preprocess_text(text) for text in test_df['full_text']]

    # TF-IDF Vectorization
    print("Creating TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )

    train_tfidf = tfidf.fit_transform(train_texts)
    test_tfidf = tfidf.transform(test_texts)

    # Combine features
    train_tfidf_df = pd.DataFrame(train_tfidf.toarray(), columns=[f'tfidf_{i}' for i in range(train_tfidf.shape[1])])
    test_tfidf_df = pd.DataFrame(test_tfidf.toarray(), columns=[f'tfidf_{i}' for i in range(test_tfidf.shape[1])])

    # Combine all features
    X_train = pd.concat([train_feat_df, train_tfidf_df], axis=1)
    X_test = pd.concat([test_feat_df, test_tfidf_df], axis=1)
    y_train = train_df['score']

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training models...")

    # Train multiple models
    models = {
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42),
        'LinearRegression': LinearRegression()
    }

    best_model = None
    best_score = float('inf')
    model_scores = {}

    # Cross-validation
    for name, model in models.items():
        print(f"Training {name}...")

        # Use appropriate data based on model type
        if name in ['Ridge', 'LinearRegression']:
            X_train_model = X_train_scaled
        else:
            X_train_model = X_train

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_model, y_train,
                                  cv=5, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-cv_scores)
        avg_rmse = np.mean(rmse_scores)

        model_scores[name] = avg_rmse
        print(f"{name} - Average RMSE: {avg_rmse:.4f} (+/- {np.std(rmse_scores):.4f})")

        if avg_rmse < best_score:
            best_score = avg_rmse
            best_model = model
            best_model_name = name

    print(f"\nBest model: {best_model_name} with RMSE: {best_score:.4f}")

    # Train the best model on full data
    if best_model_name in ['Ridge', 'LinearRegression']:
        best_model.fit(X_train_scaled, y_train)
        predictions = best_model.predict(X_test_scaled)
    else:
        best_model.fit(X_train, y_train)
        predictions = best_model.predict(X_test)

    # Ensure predictions are in valid range [1, 6]
    predictions = np.clip(predictions, 1, 6)

    # Round to nearest integer for final scores
    predictions = np.round(predictions).astype(int)

    print(f"\nPredictions: {predictions}")

    # Create submission file
    submission = pd.DataFrame({
        'essay_id': test_df['essay_id'],
        'score': predictions
    })

    submission.to_csv('submission.csv', index=False)
    print("Submission file created: submission.csv")

    return best_model, model_scores, predictions

if __name__ == "__main__":
    model, scores, predictions = train_essay_scoring_model()
