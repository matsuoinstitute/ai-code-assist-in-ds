import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Simple TF-IDF approach
print("Creating TF-IDF features...")
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
X_train = tfidf.fit_transform(train_df['full_text'])
X_test = tfidf.transform(test_df['full_text'])
y_train = train_df['score']

print(f"Feature matrix shape: {X_train.shape}")

# Train Ridge regression
print("Training Ridge regression...")
model = Ridge(alpha=1.0)
cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
rmse = np.sqrt(-cv_scores.mean())
print(f"Cross-validation RMSE: {rmse:.4f}")

# Fit and predict
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Clip and round predictions
predictions = np.clip(predictions, 1, 6)
predictions = np.round(predictions).astype(int)

print(f"Predictions: {predictions}")

# Create submission
submission = pd.DataFrame({
    'essay_id': test_df['essay_id'],
    'score': predictions
})

submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")
print("\nSubmission content:")
print(submission)
