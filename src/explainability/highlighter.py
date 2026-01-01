import re

STOPWORDS = {
    "the", "and", "or", "to", "of", "in", "a", "an", "is", "are",
    "print", "given", "find", "return", "their", "two", "sum",
    "integer", "integers", "array", "number", "numbers"
}

KEYWORD_WEIGHTS = {
    # HARD
    "negative cycle": 3.5,
    "bellman ford": 3.0,
    "dijkstra": 3.0,
    "floyd warshall": 3.0,
    "segment tree": 3.0,
    "lazy propagation": 3.5,
    "bitmask": 3.0,

    # MEDIUM
    "shortest path": 2.0,
    "graph": 1.5,
    "tree": 1.5,
    "dp": 1.5,
    "dynamic programming": 2.0,
    "bfs": 1.2,
    "dfs": 1.2,

    # SUPPORT
    "constraints": 1.0,
    "optimize": 1.0,
    "complexity": 1.0,
}

def normalize(text):
    return re.sub(r"[^a-z0-9\s]", "", text.lower())

def highlight_text(text, top_k=5):
    text = normalize(text)

    found = {}

    # Phrase matching first (important!)
    for phrase, weight in KEYWORD_WEIGHTS.items():
        if phrase in text:
            found[phrase] = weight

    # Token-level matching
    for token in text.split():
        if token in STOPWORDS:
            continue
        if token in KEYWORD_WEIGHTS:
            found[token] = max(found.get(token, 0), KEYWORD_WEIGHTS[token])

    # Sort by importance
    highlights = sorted(found.items(), key=lambda x: -x[1])

    return [k for k, _ in highlights[:top_k]] or ["Problem appears straightforward"]
