import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import textstat
import nltk
from collections import Counter

class EssayFeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=2
        )
        self.svd = TruncatedSVD(n_components=50, random_state=42)
        self.is_fitted = False
        
    def extract_basic_features(self, text):
        features = {}
        
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['paragraph_count'] = len(text.split('\n\n'))
        
        words = text.split()
        if words:
            features['avg_word_length'] = np.mean([len(word) for word in words])
            features['unique_words'] = len(set(words))
            features['lexical_diversity'] = len(set(words)) / len(words)
        else:
            features['avg_word_length'] = 0
            features['unique_words'] = 0
            features['lexical_diversity'] = 0
            
        if features['sentence_count'] > 0:
            features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
        else:
            features['avg_sentence_length'] = 0
            
        return features
    
    def extract_readability_features(self, text):
        features = {}
        
        try:
            features['flesch_kincaid_grade'] = textstat.flesch_kincaid().grade_level(text)
        except:
            features['flesch_kincaid_grade'] = 0
            
        try:
            features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
        except:
            features['flesch_reading_ease'] = 0
            
        try:
            features['coleman_liau_index'] = textstat.coleman_liau_index(text)
        except:
            features['coleman_liau_index'] = 0
            
        try:
            features['automated_readability_index'] = textstat.automated_readability_index(text)
        except:
            features['automated_readability_index'] = 0
            
        try:
            features['gunning_fog'] = textstat.gunning_fog(text)
        except:
            features['gunning_fog'] = 0
            
        try:
            features['syllable_count'] = textstat.syllable_count(text)
        except:
            features['syllable_count'] = 0
            
        return features
    
    def extract_structural_features(self, text):
        features = {}
        
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['comma_count'] = text.count(',')
        features['semicolon_count'] = text.count(';')
        features['colon_count'] = text.count(':')
        features['quote_count'] = text.count('"') + text.count("'")
        
        features['uppercase_count'] = sum(1 for c in text if c.isupper())
        features['digit_count'] = sum(1 for c in text if c.isdigit())
        
        features['capital_ratio'] = features['uppercase_count'] / len(text) if len(text) > 0 else 0
        
        return features
    
    def extract_vocabulary_features(self, text):
        features = {}
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        if words:
            word_lengths = [len(word) for word in words]
            features['min_word_length'] = min(word_lengths)
            features['max_word_length'] = max(word_lengths)
            features['std_word_length'] = np.std(word_lengths)
            
            long_words = [word for word in words if len(word) > 6]
            features['long_word_ratio'] = len(long_words) / len(words)
            
            word_freq = Counter(words)
            features['most_common_word_freq'] = word_freq.most_common(1)[0][1] if word_freq else 0
            features['hapax_legomena'] = sum(1 for count in word_freq.values() if count == 1)
            features['hapax_ratio'] = features['hapax_legomena'] / len(words)
        else:
            features['min_word_length'] = 0
            features['max_word_length'] = 0
            features['std_word_length'] = 0
            features['long_word_ratio'] = 0
            features['most_common_word_freq'] = 0
            features['hapax_legomena'] = 0
            features['hapax_ratio'] = 0
            
        return features
    
    def fit_tfidf_features(self, texts):
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.svd.fit(tfidf_matrix)
        self.is_fitted = True
        
    def extract_tfidf_features(self, texts):
        if not self.is_fitted:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf_features first.")
            
        tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        tfidf_reduced = self.svd.transform(tfidf_matrix)
        
        feature_names = [f'tfidf_component_{i}' for i in range(tfidf_reduced.shape[1])]
        return pd.DataFrame(tfidf_reduced, columns=feature_names)
    
    def extract_all_features(self, texts, fit_tfidf=False):
        if fit_tfidf:
            self.fit_tfidf_features(texts)
            
        all_features = []
        
        for text in texts:
            features = {}
            features.update(self.extract_basic_features(text))
            features.update(self.extract_readability_features(text))
            features.update(self.extract_structural_features(text))
            features.update(self.extract_vocabulary_features(text))
            all_features.append(features)
            
        features_df = pd.DataFrame(all_features)
        
        if self.is_fitted:
            tfidf_features = self.extract_tfidf_features(texts)
            features_df = pd.concat([features_df.reset_index(drop=True), 
                                   tfidf_features.reset_index(drop=True)], axis=1)
            
        features_df = features_df.fillna(0)
        
        return features_df
