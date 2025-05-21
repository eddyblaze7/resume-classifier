# train_model.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load labeled resume dataset
df = pd.read_csv('data/labeled_resume_data.csv')

# Split data
X = df['text']
y = df['label']

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_vec, y)

# Create output directory if not exists
os.makedirs('model', exist_ok=True)

# Save the model and vectorizer
joblib.dump(model, 'model/resume_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

print("✅ Model training complete. Files saved in /model")
