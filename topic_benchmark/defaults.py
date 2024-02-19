from sklearn.feature_extraction.text import CountVectorizer


def default_vectorizer():
    return CountVectorizer(min_df=10)
