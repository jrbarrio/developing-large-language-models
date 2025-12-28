# Set the number of queries
n_queries = 10
for _ in range(n_queries):
    # Use the current labeled data
    learner.teach(X_labeled, y_labeled)
    # Query from unlabeled data
    query_idx, _ = learner.query(X_unlabeled, n_instances=5)  
    X_new, y_new = X_unlabeled[query_idx], y[query_idx]  
    X_labeled = np.vstack((X_labeled, X_new))  
    y_labeled = np.append(y_labeled, y_new)  
    # Update the unlabeled dataset
    X_unlabeled = np.delete(X_unlabeled, query_idx, axis=0) 