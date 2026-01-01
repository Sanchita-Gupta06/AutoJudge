from sklearn.metrics.pairwise import cosine_similarity

def find_similar(input_vec, dataset_vecs, df, k=3):
    scores = cosine_similarity(input_vec, dataset_vecs)[0]
    idx = scores.argsort()[-k:][::-1]

    return [
        {
            "title": df.iloc[i]["title"],
            "class": df.iloc[i]["problem_class"],
            "score": df.iloc[i]["problem_score"]
        }
        for i in idx
    ]
