from typing import TypedDict

import numpy as np
from rich.progress import Progress
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer

from topic_benchmark.base import Loader
from topic_benchmark.registries import (dataset_registry, metric_registry,
                                        model_registry)


class BenchmarkEntry(TypedDict):
    dataset: str
    model: str
    results: dict[str, float]


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
    encoder, vectorizer: CountVectorizer
) -> list[BenchmarkEntry]:
    res = []
    n_datasets = len(list(dataset_registry.get_all()))
    n_models = len(list(model_registry.get_all()))
    with Progress() as progress:
        dataset_task = progress.add_task(
            "[red]Running Datasets...", total=n_datasets
        )
        model_task = progress.add_task(
            "[cyan]Running Models...", total=n_models
        )
        for dataset_name, dataset_loader in dataset_registry.get_all().items():
            progress.console.print(f"Working with {dataset_name}")
            progress.update(model_task, completed=0)
            corpus = dataset_loader()
            embeddings = encoder.encode(corpus)
            for model_name, model_loader in model_registry.get_all().items():
                progress.console.print(f"Evaluating {model_name}")
                loader = model_loader(
                    encoder=encoder, vectorizer=clone(vectorizer)
                )
                n_topics = list(range(10, 51, 10))
                scores = evaluate_model(corpus, embeddings, loader, n_topics)
                entry = BenchmarkEntry(
                    dataset=dataset_name, model=model_name, results=scores
                )
                res.append(entry)
                progress.update(model_task, advance=1)
            progress.update(dataset_task, advance=1)
    return res
