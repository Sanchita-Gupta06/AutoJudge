import re
import numpy as np

KEYWORDS = [
    "dp", "graph", "tree", "dfs", "bfs",
    "greedy", "binary search", "segment tree",
    "heap", "priority queue", "hash", "math"
]

def extract_structural_features(texts):
    features = []

    for text in texts:
        text_lower = text.lower()

        length = len(text)
        constraint_count = len(re.findall(r"10\^\d+|1e\d+", text))
        keyword_count = sum(1 for k in KEYWORDS if k in text_lower)
        has_large_constraint = int("10^5" in text or "10^6" in text)

        features.append([
            length,
            constraint_count,
            keyword_count,
            has_large_constraint
        ])

    return np.array(features)


