from sklearn.cluster import KMeans

def train_kmeans(X, n_clusters: int):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X)
    return model