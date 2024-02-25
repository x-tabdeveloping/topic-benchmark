from itertools import chain

from turftopic.data import TopicData

from topic_benchmark.base import Metric
from topic_benchmark.registries import metric_registry
from topic_benchmark.utils import get_top_k


@metric_registry.register("Diversity")
def load_diversity() -> Metric:
    top_k = 10

    def score(data: TopicData) -> float:
        topics = get_top_k(data, top_k)
        unique_words = set(chain.from_iterable(topics))
        total_words = top_k * len(topics)
        return len(unique_words) / total_words

    return score
