import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import nltk
from textstat import flesch_reading_ease, flesch_kincaid_grade, automated_readability_index
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class EssayFeatureExtractor:
    def __init__(self):
        pass

    def extract_basic_features(self, text):
        """Extract basic text features"""
        features = {}

        # Length features
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['paragraph_count'] = len(text.split('\n\n'))

        # Average lengths
        words = text.split()
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        sentences = re.split(r'[.!?]+', text)
        features['avg_sentence_length'] = np.mean([len(sent.split()) for sent in sentences if sent.strip()]) if sentences else 0

        # Punctuation features
        features['comma_count'] = text.count(',')
        features['period_count'] = text.count('.')
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['quotation_count'] = text.count('"')

        # Error indicators
        features['spelling_errors'] = len(re.findall(r'\b\w*[a-z]{2,}[A-Z]+\w*\b', text))  # Mixed case words
        features['repeated_chars'] = len(re.findall(r'(.)\1{2,}', text))  # Repeated characters

        return features

    def extract_readability_features(self, text):
        """Extract readability scores"""
        features = {}
        try:
            features['flesch_reading_ease'] = flesch_reading_ease(text)
            features['flesch_kincaid_grade'] = flesch_kincaid_grade(text)
            features['automated_readability_index'] = automated_readability_index(text)
        except:
            features['flesch_reading_ease'] = 0
            features['flesch_kincaid_grade'] = 0
            features['automated_readability_index'] = 0

        return features

    def extract_linguistic_features(self, text):
        """Extract linguistic complexity features"""
        features = {}

        # Vocabulary diversity
        words = text.lower().split()
        unique_words = set(words)
        features['vocabulary_diversity'] = len(unique_words) / len(words) if words else 0

        # Complex word ratio (words with 3+ syllables)
        complex_words = [word for word in words if self.count_syllables(word) >= 3]
        features['complex_word_ratio'] = len(complex_words) / len(words) if words else 0

        # Transition words
        transition_words = ['however', 'therefore', 'furthermore', 'moreover', 'consequently',
                          'nevertheless', 'additionally', 'meanwhile', 'subsequently']
        features['transition_word_count'] = sum(1 for word in transition_words if word in text.lower())

        return features

    def count_syllables(self, word):
        """Simple syllable counter"""
        word = word.lower()
        vowels = 'aeiouy'
        syllables = 0
        prev_char_was_vowel = False

        for char in word:
            if char in vowels:
                if not prev_char_was_vowel:
                    syllables += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False

        if word.endswith('e'):
            syllables -= 1

        return max(1, syllables)

    def extract_all_features(self, text):
        """Extract all features for a given text"""
        features = {}
        features.update(self.extract_basic_features(text))
        features.update(self.extract_readability_features(text))
        features.update(self.extract_linguistic_features(text))

        return features

class EssayScorer:
    def __init__(self):
        self.feature_extractor = EssayFeatureExtractor()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None

    def load_data(self, train_path, test_path):
        """Load training and test data"""
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)

        print(f"Training data shape: {self.train_df.shape}")
        print(f"Test data shape: {self.test_df.shape}")
        print(f"Score distribution:\n{self.train_df['score'].value_counts().sort_index()}")

    def prepare_features(self):
        """Prepare features for training and testing"""
        print("Extracting features...")

        # Extract handcrafted features
        train_features = []
        for text in self.train_df['full_text']:
            features = self.feature_extractor.extract_all_features(text)
            train_features.append(features)

        test_features = []
        for text in self.test_df['full_text']:
            features = self.feature_extractor.extract_all_features(text)
            test_features.append(features)

        # Convert to DataFrames
        self.train_features_df = pd.DataFrame(train_features)
        self.test_features_df = pd.DataFrame(test_features)

        # Fill NaN values
        self.train_features_df = self.train_features_df.fillna(0)
        self.test_features_df = self.test_features_df.fillna(0)

        # TF-IDF features
        tfidf_train = self.tfidf_vectorizer.fit_transform(self.train_df['full_text'])
        tfidf_test = self.tfidf_vectorizer.transform(self.test_df['full_text'])

        # Combine features
        from scipy.sparse import hstack
        self.X_train = hstack([
            self.scaler.fit_transform(self.train_features_df),
            tfidf_train
        ])
        self.X_test = hstack([
            self.scaler.transform(self.test_features_df),
            tfidf_test
        ])

        self.y_train = self.train_df['score'].values

        print(f"Feature matrix shape: {self.X_train.shape}")

    def train_models(self):
        """Train multiple models and find the best one"""
        print("Training models...")

        # Define models to try
        models_to_try = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Ridge': Ridge(alpha=1.0),
            'SVR': SVR(kernel='rbf', C=1.0)
        }

        best_score = float('-inf')

        for name, model in models_to_try.items():
            print(f"Training {name}...")

            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train,
                                      cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())

            print(f"{name} CV RMSE: {cv_rmse:.4f} (+/- {cv_scores.std() * 2:.4f})")

            # Train on full training set
            model.fit(self.X_train, self.y_train)
            self.models[name] = model

            # Track best model
            if -cv_scores.mean() > best_score:
                best_score = -cv_scores.mean()
                self.best_model = model
                self.best_model_name = name

        print(f"\nBest model: {self.best_model_name}")

    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning on the best model type"""
        print("Performing hyperparameter tuning...")

        if self.best_model_name == 'RandomForest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            model = RandomForestRegressor(random_state=42)

        elif self.best_model_name == 'GradientBoosting':
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            model = GradientBoostingRegressor(random_state=42)

        else:
            # Skip tuning for other models
            return

        grid_search = GridSearchCV(
            model, param_grid, cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )

        grid_search.fit(self.X_train, self.y_train)

        self.best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {np.sqrt(-grid_search.best_score_):.4f}")

    def evaluate_model(self):
        """Evaluate the best model"""
        print("Evaluating model...")

        # Split training data for evaluation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42
        )

        # Train on split and predict on validation
        self.best_model.fit(X_train_split, y_train_split)
        y_pred = self.best_model.predict(X_val_split)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val_split, y_pred))
        mae = mean_absolute_error(y_val_split, y_pred)
        r2 = r2_score(y_val_split, y_pred)

        print(f"Validation RMSE: {rmse:.4f}")
        print(f"Validation MAE: {mae:.4f}")
        print(f"Validation R²: {r2:.4f}")

        # Retrain on full training set
        self.best_model.fit(self.X_train, self.y_train)

    def predict_test(self):
        """Generate predictions for test data"""
        print("Generating predictions...")

        predictions = self.best_model.predict(self.X_test)

        # Round predictions to nearest integer and clip to valid range
        predictions = np.round(predictions).astype(int)
        predictions = np.clip(predictions, 0, 4)  # Assuming scores are 0-4

        # Create submission file
        submission = pd.DataFrame({
            'essay_id': self.test_df['essay_id'],
            'score': predictions
        })

        submission.to_csv('test_predictions.csv', index=False)
        print("Predictions saved to test_predictions.csv")

        return predictions

def main():
    # Initialize scorer
    scorer = EssayScorer()

    # Load data
    scorer.load_data('train.csv', 'test.csv')

    # Prepare features
    scorer.prepare_features()

    # Train models
    scorer.train_models()

    # Hyperparameter tuning
    scorer.hyperparameter_tuning()

    # Evaluate
    scorer.evaluate_model()

    # Generate predictions
    predictions = scorer.predict_test()

    print(f"\nPrediction distribution:\n{pd.Series(predictions).value_counts().sort_index()}")

if __name__ == "__main__":
    main()
