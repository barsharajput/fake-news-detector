import shap


def get_shap_values(model, vectorizer, text):
    explainer = shap.Explainer(model, vectorizer)
    shap_values = explainer([text])
    return shap_values
