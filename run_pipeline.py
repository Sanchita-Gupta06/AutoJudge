from src.preprocessing.load_data import load_dataset
from src.preprocessing.build_text import build_full_text
from src.preprocessing.text_cleaning import clean_text

from src.features.tfidf_vectorizer import train_tfidf
from src.features.structural_features import extract_structural_features

from src.models.train_classifier import train_classifier
from src.models.train_regressor import train_regressor
from src.models.evaluate import evaluate_classifier, evaluate_regressor

from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

df = load_dataset("data/raw/problems_data.jsonl")

df["full_text"] = df.apply(build_full_text, axis=1)
df["clean_text"] = df["full_text"].apply(clean_text)

df.to_csv("data/processed/problems_clean.csv", index=False)

# TF-IDF
X_text, vectorizer = train_tfidf(df["clean_text"])

# Structural features
X_struct = extract_structural_features(df["clean_text"])

# Combine features
X = hstack([X_text, X_struct])

X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
    X,
    df["problem_class"],
    df["problem_score"],
    test_size=0.2,
    random_state=42,
    stratify=df["problem_class"]
)

classifier = train_classifier(X_train, y_cls_train)
regressor = train_regressor(X_train, y_reg_train)

evaluate_classifier(classifier, X_test, y_cls_test)
evaluate_regressor(regressor, X_test, y_reg_test)
