# phishing_model.py
import numpy as np
import joblib

# Make sure these names match the classes you copied
from custom_transformers import TextStats, CombinedFeatures

# Load the trained pipeline
pipeline = joblib.load("phishing_pipeline.pkl")

LABELS = {
    0: "Safe Email",
    1: "Phishing Email",
}

def predict_email(text: str):
    """
    Use the trained pipeline to classify a single email.
    """
    proba = pipeline.predict_proba([text])[0]   # [P(0), P(1)]
    safe_prob = float(proba[0])
    phishing_prob = float(proba[1])

    label_idx = int(np.argmax(proba))          # 0 or 1
    label = LABELS[label_idx]

    return {
        "label": label,
        "label_idx": label_idx,
        "probabilities": {
            "safe": safe_prob,
            "phishing": phishing_prob,
        },
    }
