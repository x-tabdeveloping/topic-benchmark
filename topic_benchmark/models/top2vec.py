from functools import partial

from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from turftopic import ClusteringTopicModel
from umap.umap_ import UMAP

from topic_benchmark import model_registry
from topic_benchmark.base import Loader


@model_registry.register("Top2Vec")
def load_top2vec(encoder, vectorizer: CountVectorizer) -> Loader:
    dim_red = UMAP(n_neighbors=15, n_components=5, metric="cosine")
    clustering = HDBSCAN(
        min_cluster_size=15,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    _load = partial(
        ClusteringTopicModel,
        encoder=encoder,
        vectorizer=vectorizer,
        dimensionality_reduction=dim_red,
        clustering=clustering,
        feature_importance="centroid",
    )
    return _load
