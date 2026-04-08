from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from src.preprocess import clean_text

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
fake = pd.read_csv(DATA_DIR / "fake.csv")
real = pd.read_csv(DATA_DIR / "real.csv")
fake["label"] = 0
real["label"] = 1
data = pd.concat([fake, real]).sample(frac=1).reset_index(drop=True)

data["text"] = data["text"].apply(clean_text)
X = data["text"]
y = data["label"]

# Load model
tfidf = joblib.load(MODEL_DIR / "tfidf.pkl")
model = joblib.load(MODEL_DIR / "classifier.pkl")

X_vec = tfidf.transform(X)
y_pred = model.predict(X_vec)

# Classification report
report = classification_report(y, y_pred)
with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
    f.write(report)

print("Classification Report:\n")
print(report)

# Confusion matrix
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
disp.plot()
plt.title("Confusion Matrix - Fake News Detection")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.show()

print("\nSaved outputs to /outputs")
