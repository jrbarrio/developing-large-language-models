# Define the filter function
def filter_low_confidence_predictions(prob_dists, threshold=0.5):
    filtered_indices = [i for i, prob_dist in enumerate(prob_dists) if least_confidence(prob_dist) > threshold]
    return filtered_indices

# Find the indices
filtered_indices = filter_low_confidence_predictions(prob_dists)

high_confidence_texts = [texts[i] for i in filtered_indices]
print("High-confidence texts:", high_confidence_texts)
