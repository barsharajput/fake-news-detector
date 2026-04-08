from transformers import pipeline

# Load pre-trained model (first time will download)
classifier = pipeline("text-classification")


def predict_bert(text):
    result = classifier(text)[0]

    label = result["label"]
    score = round(result["score"] * 100, 2)

    return {"label": label, "confidence": score}
