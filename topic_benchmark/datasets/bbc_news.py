from datasets import load_dataset

from topic_benchmark.registries import dataset_registry


@dataset_registry.register("BBC News")
def load_bbc_news() -> list[str]:
    ds = load_dataset("SetFit/bbc-news", split="train")
    return ds["text"]
