from functools import partial

from sklearn.feature_extraction.text import CountVectorizer
from turftopic import (GMM, AutoEncodingTopicModel, KeyNMF,
                       SemanticSignalSeparation)

from topic_benchmark.base import Loader
from topic_benchmark.registries import model_registry


@model_registry.register("GMM")
def load_gmm(encoder, vectorizer: CountVectorizer) -> Loader:
    return partial(GMM, encoder=encoder, vectorizer=vectorizer)


@model_registry.register("KeyNMF")
def load_keynmf(encoder, vectorizer: CountVectorizer) -> Loader:
    return partial(KeyNMF, encoder=encoder, vectorizer=vectorizer)


@model_registry.register("S³")
def load_s3(encoder, vectorizer: CountVectorizer) -> Loader:
    return partial(
        SemanticSignalSeparation, encoder=encoder, vectorizer=vectorizer
    )
