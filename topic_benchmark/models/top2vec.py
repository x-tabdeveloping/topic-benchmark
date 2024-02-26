from functools import partial

from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from turftopic import ClusteringTopicModel
from umap.umap_ import UMAP

from topic_benchmark.base import Loader
from topic_benchmark.registries import model_registry


@model_registry.register("Top2Vec")
def load_top2vec(encoder, vectorizer: CountVectorizer) -> Loader:
    dim_red = UMAP(n_neighbors=15, n_components=5, metric="cosine")
    clustering = HDBSCAN(
        min_cluster_size=15,
        metric="euclidean",
        cluster_selection_method="eom",
    )

    def _load(n_components: int):
        return ClusteringTopicModel(
            encoder=encoder,
            vectorizer=vectorizer,
            dimensionality_reduction=dim_red,
            clustering=clustering,
            feature_importance="centroid",
            reduction_method="smallest",
            n_reduce_to=n_components,
        )

    return _load
