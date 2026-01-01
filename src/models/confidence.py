def confidence_and_borderline(probs):
    sorted_probs = sorted(probs, reverse=True)
    confidence = float(sorted_probs[0])  # cast to Python float
    borderline = bool((sorted_probs[0] - sorted_probs[1]) < 0.15)
    return confidence, borderline
