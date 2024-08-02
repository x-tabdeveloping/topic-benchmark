from functools import partial

from sklearn.feature_extraction.text import CountVectorizer
from turftopic import (
    GMM,
    AutoEncodingTopicModel,
    KeyNMF,
    SemanticSignalSeparation,
)

from topic_benchmark.base import Loader
from topic_benchmark.registries import model_registry


@model_registry.register("GMM")
def load_gmm(encoder, vectorizer: CountVectorizer) -> Loader:
    def _load(n_components: int, seed: int):
        return GMM(
            n_components,
            encoder=encoder,
            vectorizer=vectorizer,
            random_state=seed,
        )

    return _load


@model_registry.register("KeyNMF")
def load_keynmf(encoder, vectorizer: CountVectorizer) -> Loader:
    def _load(n_components: int, seed: int):
        return KeyNMF(
            n_components,
            encoder=encoder,
            vectorizer=vectorizer,
            random_state=seed,
        )

    return _load


@model_registry.register("S³")
def load_s3(encoder, vectorizer: CountVectorizer) -> Loader:
    def _load(n_components: int, seed: int):
        return SemanticSignalSeparation(
            n_components,
            encoder=encoder,
            vectorizer=vectorizer,
            random_state=seed,
        )

    return _load


@model_registry.register("CombinedTM")
def load_combined(encoder, vectorizer: CountVectorizer) -> Loader:
    def _load(n_components: int, seed: int):
        return AutoEncodingTopicModel(
            n_components,
            encoder=encoder,
            vectorizer=vectorizer,
            random_state=seed,
            combined=True,
        )

    return _load


@model_registry.register("ZeroShotTM")
def load_zeroshot(encoder, vectorizer: CountVectorizer) -> Loader:
    def _load(n_components: int, seed: int):
        return AutoEncodingTopicModel(
            n_components,
            encoder=encoder,
            vectorizer=vectorizer,
            random_state=seed,
            combined=False,
        )

    return _load
