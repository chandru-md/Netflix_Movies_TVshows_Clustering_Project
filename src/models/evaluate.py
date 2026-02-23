from sklearn.metrics import silhouette_score

def evaluate_model(X, labels):
    return silhouette_score(X, labels)