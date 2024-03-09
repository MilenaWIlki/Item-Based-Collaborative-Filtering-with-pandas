import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Example dataset (user-item interactions matrix)
data = {
    'User1': [1, 0, 1, 0],
    'User2': [0, 1, 0, 1],
    'User3': [1, 1, 1, 0]
}
df = pd.DataFrame(data, index=['Item1', 'Item2', 'Item3', 'Item4'])

# Calculate item-item similarity matrix
item_sim_matrix = cosine_similarity(df.T)

# Example usage:
item_idx = 0
sim_scores = list(enumerate(item_sim_matrix[item_idx]))
print("Similarity scores for item", item_idx, ":", sim_scores)
