from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import tokenize
from turftopic.data import TopicData

from topic_benchmark.base import Metric
from topic_benchmark.registries import metric_registry
from topic_benchmark.utils import get_top_k


@metric_registry.register("NPMI Coherence")
def load_npmi() -> Metric:
    top_k = 10

    def score(data: TopicData):
        topics = get_top_k(data, top_k)
        texts = [list(tokenize(text, lower=True)) for text in data["corpus"]]
        dictionary = Dictionary(texts)
        cm = CoherenceModel(
            topics=topics,
            texts=texts,
            dictionary=dictionary,
            coherence="c_npmi",
        )
        return cm.get_coherence()

    return score
