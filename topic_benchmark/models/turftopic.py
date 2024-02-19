from functools import partial

from turftopic import (GMM, AutoEncodingTopicModel, KeyNMF,
                       SemanticSignalSeparation)

from topic_benchmark import model_registry
from topic_benchmark.base import Loader


@model_registry.register("GMM")
def load_gmm() -> Loader:
    return GMM


@model_registry.register("KeyNMF")
def load_keynmf() -> Loader:
    return KeyNMF


@model_registry.register("S³")
def load_s3() -> Loader:
    return SemanticSignalSeparation


@model_registry.register("CombinedTM")
def load_combined() -> Loader:
    return partial(AutoEncodingTopicModel, combined=True)


@model_registry.register("ZeroShotTM")
def load_zeroshot() -> Loader:
    return partial(AutoEncodingTopicModel, combined=False)
