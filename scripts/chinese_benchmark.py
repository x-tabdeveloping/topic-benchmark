import itertools
from typing import Optional

import jieba
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances
from turftopic.data import TopicData

from topic_benchmark.base import Metric
from topic_benchmark.cli import run_cli
from topic_benchmark.registries import dataset_registry, metric_registry
from topic_benchmark.utils import get_top_k


def external_coherence(topics, embedding_model: SentenceTransformer):
    arrays = []
    for index, topic in enumerate(topics):
        if len(topic) > 0:
            embeddings = dict(zip(topic, embedding_model.encode(topic)))
            w1, w2 = zip(*itertools.combinations(topic, 2))
            e1 = np.stack([embeddings[w] for w in w1])
            e2 = np.stack([embeddings[w] for w in w2])
            similarities = 1 - paired_cosine_distances(e1, e2)
            arrays.append(np.nanmean(similarities))
    return np.nanmean(arrays)


@metric_registry.register("ec_ex")
def load_embedding_coherence_ex() -> Metric:
    """Multilingual external embedding coherence with a
    multilingual sentence transformer instead of a word embedding model
    """
    top_k = 10
    embedding_model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    def score(data: TopicData, dataset_name: Optional[str]):
        topics = get_top_k(data, top_k)
        return external_coherence(topics, embedding_model)

    return score


@dataset_registry.register("TNews")
def load_tnews() -> list[str]:
    """Chinese news dataset from CMTEB tokenized with Jieba"""
    ds = load_dataset("C-MTEB/TNews-classification", split="train")
    corpus = ds["text"]
    corpus = [" ".join(jieba.cut(text)) for text in corpus]
    return corpus


if __name__ == "__main__":
    run_cli(
        encoders=["paraphrase-multilingual-MiniLM-L12-v2"],
        models=[
            "BERTopic",
            "NMF",
            "LDA",
            "Top2Vec",
            "KeyNMF",
            "S³",
            "CombinedTM",
            "ZeroShotTM",
        ],
        datasets=["ArXiv ML Papers", "BBC News", "20 Newsgroups Raw", "TNews"],
        metrics=["diversity", "c_npmi", "wec_in", "ec_ex"],
        seeds=[42],
    )
