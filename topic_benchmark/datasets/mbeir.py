"""Adds all datasets from mrbench to the dataset registry"""

import numpy as np
import pandas as pd
from datasets import Image, load_dataset
from tqdm import tqdm
from turftopic.multimodal import ImageRepr

from topic_benchmark.datasets.dataset import Dataset
from topic_benchmark.registries import dataset_registry

MultimodalCorpus = tuple[list[str], list[ImageRepr]]


def get_multimodal_corpus(ds_name) -> MultimodalCorpus:
    queries = load_dataset(ds_name, "query")
    corpus = load_dataset(ds_name, "corpus")
    qrels = load_dataset(ds_name, "qrels")
    query_modality = queries["test"]["modality"][0]
    corpus_modality = corpus["corpus"]["modality"][0]
    print(ds_name, query_modality, corpus_modality)
    # We do not use datasets where the corpus doesn't contain text
    if "text" not in corpus_modality:
        raise ValueError(
            "Corpus does not contain text, can't convert to multimodal corpus."
        )
    # If the corpus is multimodal we use that
    if corpus_modality == "image,text":
        return (
            corpus["corpus"]["text"],
            corpus["corpus"]["image"],
        )
    # Otherwise we match image and text query-corpus pairs.
    # We ignore multimodal queries
    rels = (
        qrels["test"]
        .to_pandas()
        .rename(columns={"query-id": "query_id", "corpus-id": "corpus_id"})
    )
    query_ids = queries["test"]["id"]
    query_id_to_ind = pd.DataFrame(
        dict(query_id=query_ids, query_ind=np.arange(len(query_ids)))
    )
    corpus_ids = corpus["corpus"]["id"]
    corpus_id_to_ind = pd.DataFrame(
        dict(corpus_id=corpus_ids, corpus_ind=np.arange(len(corpus_ids)))
    )
    id_mapping = rels.merge(query_id_to_ind, on="query_id", how="inner").merge(
        corpus_id_to_ind, on="corpus_id", how="inner"
    )[["corpus_ind", "query_ind"]]
    queries = queries["test"].to_pandas().iloc[id_mapping["query_ind"]]
    corpus = corpus["corpus"].to_pandas().iloc[id_mapping["corpus_ind"]]
    decoder = Image(decode=True)
    if ("image" in query_modality) and (corpus_modality == "text"):
        images = [
            decoder.decode_example(im)
            for im in tqdm(queries["image"], desc="Decoding images")
        ]
        text = list(corpus["text"])
        return text, images
    if (query_modality == "text") and (corpus_modality == "image"):
        text = list(queries["text"])
        images = [
            decoder.decode_example(im)
            for im in tqdm(corpus["image"], desc="Decoding images")
        ]
        return text, images
    raise ValueError(
        "Dataset was in invalid format, could not convert to multimodal corpus."
    )


@dataset_registry.register("VisualNews")
def load_visualnews():
    text, images = get_multimodal_corpus("MRBench/mbeir_visualnews_task3")
    return Dataset(text, images=images)


@dataset_registry.register("InfoSeek")
def load_infoseek():
    text, images = get_multimodal_corpus("MRBench/mbeir_infoseek_task8")
    return Dataset(text, images=images)


@dataset_registry.register("Oven")
def load_oven():
    text, images = get_multimodal_corpus("MRBench/mbeir_oven_task8")
    return Dataset(text, images=images)


@dataset_registry.register("Edis")
def load_edis():
    text, images = get_multimodal_corpus("MRBench/mbeir_edis_task2")
    return Dataset(text, images=images)


@dataset_registry.register("WebQA")
def load_webqa():
    text, images = get_multimodal_corpus("MRBench/mbeir_webqa_task2")
    return Dataset(text, images=images)


@dataset_registry.register("Fashion200k")
def load_fashion200k():
    text, images = get_multimodal_corpus("MRBench/mbeir_fashion200k_task3")
    return Dataset(text, images=images)


@dataset_registry.register("MSCOCO")
def load_mscoco():
    text, images = get_multimodal_corpus("MRBench/mbeir_mscoco_task3")
    return Dataset(text, images=images)
