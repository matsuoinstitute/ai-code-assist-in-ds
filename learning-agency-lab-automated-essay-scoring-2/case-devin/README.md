# Automated Essay Scoring System

This repository implements an automated essay scoring system for the Learning Agency Lab competition. The system uses machine learning to predict numerical scores (1-6) for student essays.

## Overview

The system combines multiple feature extraction techniques and ensemble modeling to achieve accurate essay scoring:

- **Feature Engineering**: Extracts text statistics, readability metrics, structural features, vocabulary complexity, and TF-IDF features
- **Ensemble Modeling**: Combines Random Forest, XGBoost, and Ridge Regression with weighted averaging
- **Cross-Validation**: Uses 5-fold cross-validation for robust performance estimation

## Files

- `main.py`: Main execution script that runs the complete pipeline
- `data_processor.py`: Data loading, cleaning, and preprocessing
- `feature_engineering.py`: Feature extraction from essay text
- `model_trainer.py`: Model training, tuning, and ensemble creation
- `predictor.py`: Prediction generation and submission file creation
- `requirements.txt`: Python package dependencies

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the complete pipeline:
```bash
python main.py
```

This will:
- Load and preprocess the training and test data
- Extract comprehensive features from essay text
- Train and tune multiple models with cross-validation
- Generate predictions for test essays
- Create `test_predictions.csv` for Kaggle submission

## Features Extracted

### Basic Text Statistics
- Text length, word count, sentence count, paragraph count
- Average word length, unique words, lexical diversity
- Average sentence length

### Readability Metrics
- Flesch-Kincaid Grade Level
- Flesch Reading Ease
- Coleman-Liau Index
- Automated Readability Index
- Gunning Fog Index
- Syllable count

### Structural Features
- Punctuation usage (exclamation, question marks, commas, etc.)
- Capitalization patterns
- Quote usage

### Vocabulary Complexity
- Word length statistics
- Long word ratio
- Hapax legomena (words appearing once)
- Most common word frequency

### TF-IDF Features
- 1000 most important unigrams and bigrams
- Reduced to 50 dimensions using SVD

## Models

The system uses an ensemble of three models:

1. **Random Forest Regressor**: Handles mixed feature types well
2. **XGBoost Regressor**: Gradient boosting for high performance
3. **Ridge Regression**: Linear baseline with regularization

Models are combined using weighted averaging based on validation performance.

## Performance

The system uses cross-validation to estimate performance and calculates ensemble weights based on validation set errors. All predictions are constrained to the valid score range [1, 6].

## Output

The system generates `test_predictions.csv` with the required format:
```
essay_id,score
000d118,3
000fe60,4
001ab80,4
```
