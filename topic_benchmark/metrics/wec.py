import itertools

import gensim.downloader as api
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
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
            arrays.append(np.nanmean(local_simi))
    return np.nanmean(arrays)


@metric_registry.register("Word Embedding Coherence")
def load_wec() -> Metric:
    top_k = 10
    wv = api.load("word2vec-google-news-300")

    def score(data: TopicData):
        topics = get_top_k(data, top_k)
        return word_embedding_coherence(topics, wv)

    return score


@metric_registry.register("IWEC")
def load_iwec() -> Metric:
    """Internal word embedding coherence:
    Trains word2vec model on the corpus, then uses it to evaluate
    based on WEC."""
    top_k = 10

    def score(data: TopicData):
        tokenizer = CountVectorizer(vocabulary=data["vocab"]).build_analyzer()
        texts = [tokenizer(text) for text in data["corpus"]]
        model = Word2Vec(texts, min_count=1)
        topics = get_top_k(data, top_k)
        return word_embedding_coherence(topics, model.wv)

    return score
