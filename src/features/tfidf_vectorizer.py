from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def train_tfidf(texts):
    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        max_features=30000,
        min_df=3
    )
    X = vectorizer.fit_transform(texts)
    joblib.dump(vectorizer, "models_saved/tfidf.pkl")
    return X, vectorizer
