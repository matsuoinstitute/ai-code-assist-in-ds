import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        
    def load_data(self, train_path='train.csv', test_path='test.csv'):
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        
        print(f"Loaded training data: {self.train_data.shape}")
        print(f"Loaded test data: {self.test_data.shape}")
        
        return self.train_data, self.test_data
    
    def clean_text(self, text):
        if pd.isna(text):
            return ""
            
        text = str(text)
        
        text = re.sub(r'\s+', ' ', text)
        
        text = text.strip()
        
        return text
    
    def preprocess_data(self):
        if self.train_data is None or self.test_data is None:
            raise ValueError("Data not loaded. Call load_data first.")
            
        self.train_data['full_text'] = self.train_data['full_text'].apply(self.clean_text)
        self.test_data['full_text'] = self.test_data['full_text'].apply(self.clean_text)
        
        print("Text preprocessing completed")
        
    def create_train_val_split(self, test_size=0.2, random_state=42):
        if self.train_data is None:
            raise ValueError("Training data not loaded.")
            
        X = self.train_data['full_text']
        y = self.train_data['score']
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        
        return X_train, X_val, y_train, y_val
    
    def get_score_distribution(self):
        if self.train_data is None:
            raise ValueError("Training data not loaded.")
            
        score_dist = self.train_data['score'].value_counts().sort_index()
        print("Score distribution:")
        print(score_dist)
        
        return score_dist
    
    def get_text_statistics(self):
        if self.train_data is None:
            raise ValueError("Training data not loaded.")
            
        self.train_data['text_length'] = self.train_data['full_text'].str.len()
        
        print("Text length statistics:")
        print(self.train_data['text_length'].describe())
        
        return self.train_data['text_length'].describe()
