import io
import time
from collections import namedtuple
from contextlib import redirect_stdout
from typing import Iterable, Optional, TypedDict, Union

from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer

from topic_benchmark.base import Loader, TopicModel
from topic_benchmark.registries import (dataset_registry, metric_registry,
                                        model_registry)
from topic_benchmark.utils import get_top_k


class BenchmarkEntry(TypedDict):
    dataset: str
    model: str
    n_topics: int
    seed: int
    topic_descriptions: list[list[str]]
    runtime_s: float
    results: dict[str, float]


class BenchmarkError(TypedDict):
    dataset: str
    model: str
    n_topics: int
    seed: int
    error_message: str


EntryID = namedtuple("EntryID", ["dataset", "model", "n_topics", "seed"])


def get_entry_id(entry: Union[BenchmarkError, BenchmarkEntry]) -> EntryID:
    return EntryID(
        entry["dataset"], entry["model"], entry["n_topics"], entry["seed"]
    )


def evaluate_topics(
    topic_data,
    metrics: Optional[list[str]] = None,
    dataset_name: Optional[str] = None,
) -> dict[str, float]:
    res = {}
    for metric_name, metric_loader in metric_registry.get_all().items():
        print(f"            - Evaluating on {metric_name}")
        if (metrics is not None) and (metric_name not in metrics):
            continue
        metric = metric_loader()
        score = metric(topic_data, dataset_name=dataset_name)
        res[metric_name] = float(score)
    return res


def run_benchmark(
    encoder,
    vectorizer: CountVectorizer,
    models: Optional[list[str]] = None,
    datasets: Optional[list[str]] = None,
    metrics: Optional[list[str]] = None,
    seeds: tuple[int] = (42),
    prev_entries: Iterable[Union[BenchmarkEntry, BenchmarkError]] = (),
) -> Iterable[Union[BenchmarkEntry, BenchmarkError]]:
    done = set([get_entry_id(entry) for entry in prev_entries])
    for dataset_name, dataset_loader in dataset_registry.get_all().items():
        if (datasets is not None) and (dataset_name not in datasets):
            continue
        print(f"Evaluating models on {dataset_name}")
        print("....................................")
        corpus = dataset_loader()
        embeddings = encoder.encode(corpus)
        for model_name, model_loader in model_registry.get_all().items():
            print("   -------------------------")
            print(f"   |Evaluating {model_name}|")
            print("   _________________________")
            if (models is not None) and (model_name not in models):
                continue
            loader = model_loader(
                encoder=encoder, vectorizer=clone(vectorizer)
            )
            n_topics = list(range(10, 51, 10))
            for n_components in n_topics:
                print(f"    - {n_components} topics")
                for seed in seeds:
                    print(f"      - Seed: {seed}")
                    current_id = EntryID(
                        dataset=dataset_name,
                        model=model_name,
                        seed=seed,
                        n_topics=n_components,
                    )
                    if current_id in done:
                        print(
                            f"         Entry {current_id} already completed, skipping."
                        )
                        continue
                    model = loader(n_components=n_components, seed=seed)
                    try:
                        start_time = time.time()
                        faux_stdout = io.StringIO()
                        with redirect_stdout(faux_stdout):
                            topic_data = model.prepare_topic_data(
                                corpus, embeddings
                            )
                        end_time = time.time()
                        topic_descriptions = get_top_k(topic_data, top_k=10)
                        res = evaluate_topics(
                            topic_data,
                            metrics=metrics,
                            dataset_name=dataset_name,
                        )
                        yield BenchmarkEntry(
                            dataset=dataset_name,
                            model=model_name,
                            seed=seed,
                            n_topics=n_components,
                            topic_descriptions=topic_descriptions,
                            runtime_s=end_time - start_time,
                            results=res,
                        )
                    except Exception as e:
                        yield BenchmarkError(
                            dataset=dataset_name,
                            seed=seed,
                            model=model_name,
                            error_message=str(e),
                            n_topics=n_components,
                        )
