import pandas as pd
import numpy as np
import nltk
import warnings
warnings.filterwarnings('ignore')

from data_processor import DataProcessor
from feature_engineering import EssayFeatureExtractor
from model_trainer import EssayScorePredictor
from predictor import EssayPredictor

def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("NLTK data downloaded successfully")
    except Exception as e:
        print(f"Warning: Could not download NLTK data: {e}")

def main():
    print("=== Automated Essay Scoring System ===")
    
    download_nltk_data()
    
    print("\n1. Loading and preprocessing data...")
    processor = DataProcessor()
    train_data, test_data = processor.load_data()
    processor.preprocess_data()
    
    print("\n2. Data analysis...")
    processor.get_score_distribution()
    processor.get_text_statistics()
    
    print("\n3. Creating train/validation split...")
    X_train, X_val, y_train, y_val = processor.create_train_val_split()
    
    print("\n4. Extracting features...")
    feature_extractor = EssayFeatureExtractor()
    
    print("  - Training set features...")
    train_features = feature_extractor.extract_all_features(X_train.tolist(), fit_tfidf=True)
    
    print("  - Validation set features...")
    val_features = feature_extractor.extract_all_features(X_val.tolist())
    
    print("  - Test set features...")
    test_features = feature_extractor.extract_all_features(test_data['full_text'].tolist())
    
    print(f"Feature matrix shape: {train_features.shape}")
    print(f"Number of features: {len(train_features.columns)}")
    
    print("\n5. Training models...")
    model = EssayScorePredictor()
    model.train_models(train_features, y_train, tune_params=False)
    
    print("\n6. Cross-validation...")
    cv_scores = model.cross_validate_models(train_features, y_train)
    
    print("\n7. Calculating ensemble weights...")
    model.calculate_ensemble_weights(val_features, y_val)
    
    print("\n8. Evaluating on validation set...")
    val_predictions = model.evaluate_models(val_features, y_val)
    
    print("\n9. Generating test predictions...")
    test_predictions, individual_test_preds = model.predict(test_features)
    
    print("\n10. Creating submission file...")
    predictor = EssayPredictor()
    predictor.feature_extractor = feature_extractor
    predictor.model = model
    
    submission_df = predictor.create_submission_file(test_data, test_predictions)
    
    print("\n11. Validating predictions...")
    predictor.validate_predictions(test_predictions)
    
    print("\n12. Saving models...")
    model.save_models('essay_scorer')
    
    print("\n=== Results Summary ===")
    print(f"Training samples: {len(train_features)}")
    print(f"Validation samples: {len(val_features)}")
    print(f"Test samples: {len(test_features)}")
    print(f"Features extracted: {len(train_features.columns)}")
    
    print("\nCross-validation scores:")
    for model_name, scores in cv_scores.items():
        print(f"  {model_name}: {scores['mean_rmse']:.4f} (+/- {scores['std_rmse']:.4f}) RMSE")
    
    print(f"\nValidation RMSE: {np.sqrt(np.mean((y_val - val_predictions) ** 2)):.4f}")
    
    print("\nTest predictions:")
    print(submission_df)
    
    print("\n=== Essay Scoring Complete ===")
    print("Submission file: test_predictions.csv")

if __name__ == "__main__":
    main()
