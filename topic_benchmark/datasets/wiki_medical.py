from datasets import load_dataset

from topic_benchmark.registries import dataset_registry


@dataset_registry.register("Wiki Medical")
def load_wiki_medical() -> list[str]:
    ds = load_dataset("gamino/wiki_medical_terms", split="train")
    return ds["page_text"]
