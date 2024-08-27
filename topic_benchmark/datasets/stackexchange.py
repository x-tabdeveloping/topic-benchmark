import itertools
from datasets import load_dataset

from topic_benchmark.registries import dataset_registry


@dataset_registry.register("StackExchange")
def load_wiki_medical() -> list[str]:
    ds = load_dataset("mteb/stackexchange-clustering-p2p", split="test")
    return list(itertools.chain.from_iterable(ds["sentences"]))
