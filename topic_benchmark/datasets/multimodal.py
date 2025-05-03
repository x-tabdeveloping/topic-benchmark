"""Adds all datasets from mrbench to the dataset registry"""

import numpy as np
import pandas as pd
from datasets import load_dataset
from turftopic.multimodal import ImageRepr

from topic_benchmark.datasets.dataset import Dataset
from topic_benchmark.registries import dataset_registry

MRBENCH_DATASETS = [
    "MRBench/mbeir_visualnes_task3",
    "MRBench/mbeir_visualnes_task0",
    "MRBench/mbeir_infoseek_task8",
    "MRBench/mbeir_infoseek_task6",
    "MRBench/mbeir_oven_task8",
    "MRBench/mbeir_oven_task6",
    "MRBench/mbeir_nights_task4",
    "MRBench/mbeir_edis_task2",
    "MRBench/mbeir_webqa_task2",
    "MRBench/mbeir_webqa_task1",
    "MRBench/mbeir_fashioniq_task7",
    "MRBench/mbeir_fashion200k_task3",
    "MRBench/mbeir_fashion200k_task0",
    "MRBench/mbeir_cirr_task7",
    "MRBench/mbeir_mscoco_task3",
    "MRBench/mbeir_mscoco_task0",
]

MultimodalCorpus = tuple[list[str], list[ImageRepr]]


def get_multimodal_corpus(dataset) -> MultimodalCorpus:
    query_modality = dataset["query"]["test"]["modality"][0]
    corpus_modality = dataset["corpus"]["corpus"]["modality"][0]
    # We do not use datasets where the corpus doesn't contain text
    if "text" not in corpus_modality:
        raise ValueError(
            "Corpus does not contain text, can't convert to multimodal corpus."
        )
    # If the corpus is multimodal we use that
    if corpus_modality == "image,text":
        return (
            dataset["corpus"]["corpus"]["text"],
            dataset["corpus"]["corpus"]["image"],
        )
    # Otherwise we match image and text query-corpus pairs.
    # We ignore multimodal queries
    rels = dataset["qrels"]["test"].to_pandas()
    query_ids = dataset["query"]["test"]["id"]
    query_id_to_ind = pd.DataFrame(
        dict(query_id=query_ids, query_ind=np.arange(len(query_ids)))
    )
    corpus_ids = dataset["corpus"]["corpus"]["id"]
    corpus_id_to_ind = pd.DataFrame(
        dict(corpus_id=corpus_ids, corpus_ind=np.arange(len(corpus_ids)))
    )
    id_mapping = rels.merge(
        query_id_to_ind, on="corpus_id", how="inner"
    ).merge(corpus_id_to_ind, on="corpus_id", how="inner")[
        ["corpus_ind", "query_ind"]
    ]
    if ("image" in query_modality) and (corpus_modality == "text"):
        images = dataset["query"]["test"]["image"].select(
            id_mapping["query_ind"]
        )
        text = dataset["corpus"]["corpus"]["text"].select(
            id_mapping["corpus_ind"]
        )
        return text, images
    if (query_modality == "text") and (corpus_modality == "image"):
        images = dataset["query"]["test"]["image"].select(
            id_mapping["query_ind"]
        )
        text = dataset["corpus"]["corpus"]["text"].select(
            id_mapping["corpus_ind"]
        )
        return text, images
    raise ValueError(
        "Dataset was in invalid format, could not convert to multimodal corpus."
    )


for ds_name in MRBENCH_DATASETS:

    def _load_dataset():
        ds = load_dataset(ds_name)
        text, images = get_multimodal_corpus(ds)
        return Dataset(text, images=images)

    dataset_registry.register(ds_name.split("/")[-1], _load_dataset)
