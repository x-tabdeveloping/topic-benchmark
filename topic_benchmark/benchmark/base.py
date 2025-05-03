from collections import namedtuple
from typing import TypedDict


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
