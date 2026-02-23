import joblib
from src.data.load_data import load_data
from src.data.preprocess import clean_data
from src.features.text_features import create_text_column, tfidf_vectorize, reduce_dimensions
from src.models.train_kmeans import train_kmeans
from src.models.evaluate import evaluate_model
from src.models.interpret import get_cluster_summary

def run_pipeline(data_path, content_type="Movie", n_clusters=10):

    df = load_data(data_path)
    df = clean_data(df)

    df = df[df["type"] == content_type].copy()

    text_data = create_text_column(df)

    X, vectorizer = tfidf_vectorize(text_data)
    X_reduced, svd = reduce_dimensions(X)

    model = train_kmeans(X_reduced, n_clusters=n_clusters)

    labels = model.labels_
    score = evaluate_model(X_reduced, labels)

    print(f"Silhouette Score: {score}")

    joblib.dump(model, f"models/{content_type.lower()}_kmeans.pkl")

    get_cluster_summary(movies_data_cleaned)
    get_cluster_summary(tv_data_cleaned)

    return df, model