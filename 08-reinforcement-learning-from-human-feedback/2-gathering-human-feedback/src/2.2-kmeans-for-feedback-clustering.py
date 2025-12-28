def detect_anomalies(data, n_clusters=3):
    # Initialize k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_

    # Calculate distances from cluster centers
    distances = np.linalg.norm(data - centers[clusters], axis=1)
    return distances
  
anomalies = detect_anomalies(confidences)
print(anomalies)