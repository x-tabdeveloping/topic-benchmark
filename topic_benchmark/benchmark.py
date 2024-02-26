from typing import Iterable, TypedDict, Union

import numpy as np
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer

from topic_benchmark.base import Loader
from topic_benchmark.registries import (
    dataset_registry,
    metric_registry,
    model_registry,
)


class BenchmarkEntry(TypedDict):
    dataset: str
    model: str
    results: dict[str, float]


class BenchmarkError(TypedDict):
    dataset: str
    model: str
    error_message: str


def evaluate_model(
    corpus,
    embeddings,
    loader: Loader,
    n_topics: list[int],
) -> dict[str, float]:
    res = {}
    for n_components in n_topics:
        model = loader(n_components=n_components)
        topic_data = model.prepare_topic_data(corpus, embeddings)
        for metric_name, metric_loader in metric_registry.get_all().items():
            if metric_name not in res:
                res[metric_name] = []
            metric = metric_loader()
            score = metric(topic_data)
            res[metric_name].append(score)
    return {
        metric_name: float(np.mean(scores))
        for metric_name, scores in res.items()
    }


def run_benchmark(
    encoder, vectorizer: CountVectorizer, done: set[tuple[str, str]]
) -> Iterable[Union[BenchmarkEntry, BenchmarkError]]:
    for dataset_name, dataset_loader in dataset_registry.get_all().items():
        print(f"Evaluating models on {dataset_name}")
        corpus = dataset_loader()
        embeddings = encoder.encode(corpus)
        for model_name, model_loader in model_registry.get_all().items():
            if (dataset_name, model_name) in done:
                print(f"Model {model_name} already done, skipping")
                continue
            print(f"Evaluating {model_name}")
            try:
                loader = model_loader(
                    encoder=encoder, vectorizer=clone(vectorizer)
                )
                n_topics = list(range(10, 51, 10))
                scores = evaluate_model(corpus, embeddings, loader, n_topics)
                entry = BenchmarkEntry(
                    dataset=dataset_name, model=model_name, results=scores
                )
            except Exception as e:
                entry = BenchmarkError(
                    dataset=dataset_name,
                    model=model_name,
                    error_message=str(e),
                )
            yield entry
