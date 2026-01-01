from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

def evaluate_classifier(model, X_test, y_test):
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    cm = confusion_matrix(y_test, preds)
    cm_df = pd.DataFrame(cm)

    
    print("\n CLASSIFICATION METRICS")
    print("\nConfusion Matrix:")
    print(cm_df)
    print(f"Accuracy : {acc:.3f}")
    print(f"F1 Score : {f1:.3f}")

    return acc, f1


def evaluate_regressor(model, X_test, y_test):
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    max_err = np.max(np.abs(y_test - preds))

    
    print("\n REGRESSION METRICS")
    print(f"Max Error : {max_err:.3f}")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")

    return mae, rmse, r2
