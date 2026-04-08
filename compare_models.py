from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.preprocess import clean_text

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
fake = pd.read_csv(DATA_DIR / "fake.csv")
real = pd.read_csv(DATA_DIR / "real.csv")

fake["label"] = 0
real["label"] = 1

data = pd.concat([fake, real]).sample(frac=1).reset_index(drop=True)
data["text"] = data["text"].apply(clean_text)

X = data["text"]
y = data["label"]

# -----------------------------
# Vectorization
# -----------------------------
tfidf = TfidfVectorizer(max_features=5000)
X_vec = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# -----------------------------
# Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
}

results = {}

# -----------------------------
# Train & Evaluate
# -----------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    results[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1 Score": f1_score(y_test, preds),
    }

# -----------------------------
# Print Results
# -----------------------------
print("\nMODEL COMPARISON RESULTS\n")
for model, metrics in results.items():
    print(model)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()

# -----------------------------
# Plot Comparison
# -----------------------------
df = pd.DataFrame(results).T
df.plot(kind="bar", figsize=(10, 6))
plt.title("Model Comparison: Logistic Regression vs Naive Bayes")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png")
plt.show()
