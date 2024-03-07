import time
from typing import Iterable, TypedDict, Union

from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer

from topic_benchmark.base import Loader
from topic_benchmark.registries import dataset_registry, metric_registry, model_registry
from topic_benchmark.utils import get_top_k


class BenchmarkEntry(TypedDict):
    dataset: str
    model: str
    n_topics: int
    topic_descriptions: list[list[str]]
    runtime_s: float
    results: dict[str, float]


class BenchmarkError(TypedDict):
    dataset: str
    model: str
    n_topics: int
    error_message: str


def evaluate_model(
    dataset_name: str,
    model_name: str,
    corpus,
    embeddings,
    loader: Loader,
    n_topics: list[int],
    done: set[tuple[str, str, int]],
) -> Iterable[Union[BenchmarkEntry, BenchmarkError]]:
    for n_components in n_topics:
        print(f" - Evaluating on {n_components} topics")
        if (dataset_name, model_name, n_components) in done:
            print(f"Model {model_name}({n_components}) already done, skipping")
            continue
        model = loader(n_components=n_components)
        try:
            start_time = time.time()
            topic_data = model.prepare_topic_data(corpus, embeddings)
            end_time = time.time()
        except Exception as e:
            yield BenchmarkError(
                dataset=dataset_name,
                model=model_name,
                error_message=str(e),
                n_topics=n_components,
            )
            continue
        topic_descriptions = get_top_k(topic_data, top_k=10)
        res = {}
        for metric_name, metric_loader in metric_registry.get_all().items():
            metric = metric_loader()
            score = metric(topic_data)
            res[metric_name] = float(score)
        yield BenchmarkEntry(
            dataset=dataset_name,
            model=model_name,
            n_topics=n_components,
            topic_descriptions=topic_descriptions,
            runtime_s=end_time - start_time,
            results=res,
        )


def run_benchmark(
    encoder, vectorizer: CountVectorizer, done: set[tuple[str, str, int]]
) -> Iterable[Union[BenchmarkEntry, BenchmarkError]]:
    for dataset_name, dataset_loader in dataset_registry.get_all().items():
        print(f"Evaluating models on {dataset_name}")
        corpus = dataset_loader()
        embeddings = encoder.encode(corpus)
        for model_name, model_loader in model_registry.get_all().items():
            print(f"Evaluating {model_name}")
            loader = model_loader(encoder=encoder, vectorizer=clone(vectorizer))
            n_topics = list(range(10, 51, 10))
            entries = evaluate_model(
                dataset_name=dataset_name,
                model_name=model_name,
                corpus=corpus,
                embeddings=embeddings,
                loader=loader,
                n_topics=n_topics,
                done=done,
            )
            for entry in entries:
                yield entry
