def detect_unreliable_source(df):
    df_majority = df.groupby('id').apply(majority_vote)
    disagreements = {source: 0 for source in df['source'].unique()}
    for _, row in df.iterrows():
        # Condition to find a disagreement with majority vote
        if (row['chosen'], row['rejected']) != df_majority[row['id']]:
            disagreements[row['source']] += 1
    unreliable_source = max(disagreements, key=disagreements.get)
    return unreliable_source

disagreement = detect_unreliable_source(automotive_df)
print("Unreliable Source:", disagreement)