import pandas as pd
import numpy as np
from feature_engineering import EssayFeatureExtractor
from model_trainer import EssayScorePredictor

class EssayPredictor:
    def __init__(self):
        self.feature_extractor = EssayFeatureExtractor()
        self.model = EssayScorePredictor()
        
    def load_trained_model(self, model_prefix='essay_scorer'):
        self.model.load_models(model_prefix)
        
    def predict_scores(self, test_texts, feature_extractor=None):
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
            
        if not self.feature_extractor.is_fitted:
            raise ValueError("Feature extractor not fitted. Train the model first.")
            
        test_features = self.feature_extractor.extract_all_features(test_texts)
        
        predictions, individual_preds = self.model.predict(test_features)
        
        return predictions
    
    def create_submission_file(self, test_df, predictions, output_path='test_predictions.csv'):
        submission_df = pd.DataFrame({
            'essay_id': test_df['essay_id'],
            'score': predictions
        })
        
        submission_df.to_csv(output_path, index=False)
        print(f"Submission file saved to: {output_path}")
        
        return submission_df
    
    def validate_predictions(self, predictions):
        if not all(isinstance(p, (int, np.integer)) for p in predictions):
            print("Warning: Not all predictions are integers")
            
        if not all(1 <= p <= 6 for p in predictions):
            print("Warning: Some predictions are outside the valid range [1, 6]")
            
        print(f"Prediction distribution:")
        unique, counts = np.unique(predictions, return_counts=True)
        for score, count in zip(unique, counts):
            print(f"  Score {score}: {count} essays")
            
        return True
