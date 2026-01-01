import pandas as pd

def load_dataset(path):
    return pd.read_json(path, lines=True)