from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import os
import joblib

def train_regressor(X, y):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "ridge": Ridge(alpha=1.0),
        "rf": RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
    }

    best_model = None
    best_mae = float("inf")

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        print(f"{name} MAE:", mae)

        if mae < best_mae:
            best_mae = mae
            best_model = model

    print("Best regressor selected:", type(best_model).__name__)

    SAVE_DIR = "models_saved"
    os.makedirs(SAVE_DIR, exist_ok=True)
    joblib.dump(best_model, f"{SAVE_DIR}/regressor.pkl")

    return best_model
