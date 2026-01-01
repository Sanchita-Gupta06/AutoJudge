import joblib
import pandas as pd
from scipy.sparse import hstack

from src.explainability.highlighter import highlight_text
from src.models.similarity_engine import find_similar
from src.models.confidence import confidence_and_borderline
from src.features.structural_features import extract_structural_features
from src.preprocessing.text_cleaning import clean_text

# Load Artifacts
clf = joblib.load("models_saved/classifier.pkl")
reg = joblib.load("models_saved/regressor.pkl")
tfidf = joblib.load("models_saved/tfidf.pkl")

# Dataset for similarity search
df = pd.read_csv("data/processed/problems_clean.csv")
dataset_text_vecs = tfidf.transform(df["clean_text"])
dataset_struct_vecs = extract_structural_features(df["clean_text"])
dataset_vecs = hstack([dataset_text_vecs, dataset_struct_vecs])


# Prediction
def predict(payload: dict):

    #  Handle empty input
    if not any(v.strip() for v in payload.values()):
        return {
            "error": "Please enter at least one field to analyze difficulty."
        }

    # Combine and clean input
    raw_text = " ".join(payload.values()).lower()
    cleaned = clean_text(raw_text)

    # Features
    X_text = tfidf.transform([cleaned])
    X_struct = extract_structural_features([cleaned])
    X = hstack([X_text, X_struct])

    # Classification
    class_probs = clf.predict_proba(X)[0]
    pred_class = clf.classes_[class_probs.argmax()]
    confidence, borderline = confidence_and_borderline(class_probs)

    # Regression
    score = float(round(reg.predict(X)[0], 2))

    # Explainability
    highlights = highlight_text(cleaned)
    if len(highlights) == 0:
        highlights = ["This problem is straightforward and requires basic implementation."]

    # Similarity
    similar = find_similar(X, dataset_vecs, df)

    return {
        "predicted_class": str(pred_class),
        "predicted_score": float(round(score, 2)),
        "confidence": float(round(confidence, 2)),
        "borderline": bool(borderline),
        "highlights": highlight_text(cleaned),
        "similar_problems": find_similar(X, dataset_vecs, df)
    }
