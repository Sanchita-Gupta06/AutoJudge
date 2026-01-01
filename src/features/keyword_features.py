KEYWORDS = {
    "easy": ["array", "loop", "print"],
    "medium": ["dp", "greedy", "binary"],
    "hard": ["graph", "tree", "dfs", "bfs", "segment"]
}

def extract_keyword_features(text):
    features = []
    for group in KEYWORDS.values():
        features.append(sum(text.count(word) for word in group))
    return features
