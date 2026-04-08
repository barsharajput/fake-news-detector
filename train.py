from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.preprocess import clean_text

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# Load data
fake = pd.read_csv(DATA_DIR / "fake.csv")
real = pd.read_csv(DATA_DIR / "real.csv")

fake["label"] = 0
real["label"] = 1

data = pd.concat([fake, real]).sample(frac=1).reset_index(drop=True)
data["text"] = data["text"].apply(clean_text)

X = data["text"]
y = data["label"]

# Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_vec = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_model.predict(X_test)))

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_model.predict(X_test)))

# Save models
joblib.dump(tfidf, MODEL_DIR / "tfidf.pkl")
joblib.dump(lr_model, MODEL_DIR / "classifier.pkl")
joblib.dump(nb_model, MODEL_DIR / "naive_bayes.pkl")

print("Both models saved successfully!")
