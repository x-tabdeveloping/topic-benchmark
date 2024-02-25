import itertools

import gensim.downloader as api
import numpy as np
from turftopic.data import TopicData

from topic_benchmark.base import Metric
from topic_benchmark.registries import metric_registry
from topic_benchmark.utils import get_top_k


def word_embedding_coherence(topics, wv):
    arrays = []
    for index, topic in enumerate(topics):
        if len(topic) > 0:
            local_simi = []
            for word1, word2 in itertools.combinations(topic, 2):
                if word1 in wv.index_to_key and word2 in wv.index_to_key:
                    local_simi.append(wv.similarity(word1, word2))
            arrays.append(np.mean(local_simi))
    return np.mean(arrays)


@metric_registry.register("Word Embedding Coherence")
def load_wec() -> Metric:
    top_k = 10
    wv = api.load("word2vec-google-news-300")

    def score(data: TopicData):
        topics = get_top_k(data, top_k)
        return word_embedding_coherence(topics, wv)

    return score
