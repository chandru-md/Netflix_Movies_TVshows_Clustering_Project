from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def create_text_column(df):
    return df["description"] + " " + df["listed_in"]

def tfidf_vectorize(text_data):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1,2)
    )
    X = vectorizer.fit_transform(text_data)
    return X, vectorizer

def reduce_dimensions(X, n_components=100):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)
    return X_reduced, svd