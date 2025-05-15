import itertools
from typing import Optional

from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from turftopic.data import TopicData

from topic_benchmark.base import Metric
from topic_benchmark.registries import metric_registry

STOPS = set(ENGLISH_STOP_WORDS)


def stop_word_rel_freq(topic_descriptions: list[list[str]]) -> float:
    words = list(itertools.chain.from_iterable(topic_descriptions))
    total = len(words)
    n_stop = 0
    for word in words:
        if word in STOPS:
            n_stop += 1
    return n_stop / total


@metric_registry.register("stop_freq")
def load_stop_freq(top_k: int = 10) -> Metric:

    def score(data: TopicData, dataset_name: Optional[str]):
        top_words = data.get_top_words(top_k=top_k)
        return stop_word_rel_freq(top_words)

    return score
