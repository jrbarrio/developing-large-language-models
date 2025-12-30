from collections import Counter

def majority_vote(df):
  	# Count occurrences of each (chosen, rejected) pair
    votes = Counter(zip(df['chosen'], df['rejected']))
    # Find the (chosen, rejected) pair with the highest vote count
    winner = max(votes, key=votes.get)
    return winner

final_preferences = quality_df.groupby(['id']).apply(majority_vote)

print(final_preferences)