from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import joblib
import os

def train_classifier(X, y):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logreg": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            n_jobs=-1),
        "svm": LinearSVC(max_iter=5000)
    }

    best_model = None
    best_f1 = 0

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds, average="macro")
        print(f"{name} F1:", f1)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    print("Best classifier selected:", type(best_model).__name__)

    os.makedirs("models_saved", exist_ok=True)
    joblib.dump(best_model, "models_saved/classifier.pkl")

    return best_model
