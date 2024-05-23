from datasets import load_dataset

from topic_benchmark.registries import dataset_registry


@dataset_registry.register("ArXiv ML Papers")
def load_arxiv_ml() -> list[str]:
    ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
    ds = ds.train_test_split(test_size=2048, seed=42)["test"]
    return ds["abstract"]
