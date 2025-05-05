import io
import time
from collections import namedtuple
from contextlib import redirect_stdout
from typing import Iterable, Optional, TypedDict, Union

import numpy as np
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer
from turftopic.multimodal import (ImageRepr, _load_images,
                                  _naive_join_embeddings)

from topic_benchmark.base import (BenchmarkEntry, BenchmarkError, EntryID,
                                  Loader, TopicModel)
from topic_benchmark.registries import (dataset_registry, metric_registry,
                                        model_registry)


def encode_multimodal(
    encoder,
    sentences: list[str],
    images: list[ImageRepr],
) -> dict[str, np.ndarray]:
    """Produce multimodal embeddings of the documents passed to the model."""
    if len(sentences) != len(images):
        raise ValueError("Images and documents were not the same length.")
    if hasattr(encoder, "get_text_embeddings"):
        text_embeddings = np.array(encoder.get_text_embeddings(sentences))
    else:
        text_embeddings = encoder.encode(sentences)
    embedding_size = text_embeddings.shape[1]
    images = _load_images(images)
    if hasattr(encoder, "get_image_embeddings"):
        image_embeddings = np.array(encoder.get_image_embeddings(list(images)))
    else:
        image_embeddings = []
        for image in images:
            if image is not None:
                image_embeddings.append(encoder.encode(image))
            else:
                image_embeddings.append(np.full(embedding_size, np.nan))
        image_embeddings = np.stack(image_embeddings)
        print(image_embeddings)
    if hasattr(encoder, "get_fused_embeddings"):
        document_embeddings = np.array(
            encoder.get_fused_embeddings(
                texts=sentences,
                images=list(images),
            )
        )
    else:
        document_embeddings = _naive_join_embeddings(
            text_embeddings, image_embeddings
        )

    return {
        "text_embeddings": text_embeddings,
        "image_embeddings": image_embeddings,
        "document_embeddings": document_embeddings,
    }


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
    prev_entries: Iterable[BenchmarkEntry] = (),
    multimodal: bool = False,
) -> Iterable[BenchmarkEntry]:
    done = set([entry.entry_id for entry in prev_entries])
    for dataset_name, dataset_loader in dataset_registry.get_all().items():
        if (datasets is not None) and (dataset_name not in datasets):
            continue
        print(f"Evaluating models on {dataset_name}")
        print("....................................")
        corpus = dataset_loader()
        if multimodal:
            if not corpus.images:
                print("Corpus is not multimodal, skipping...")
                continue
            embeddings = encode_multimodal(
                encoder=encoder, sentences=corpus.texts, images=corpus.images
            )
        else:
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
                            if multimodal:
                                topic_data = (
                                    model.prepare_multimodal_topic_data(
                                        corpus.texts,
                                        images=corpus.images,
                                        embeddings=embeddings,
                                    )
                                )
                            else:
                                topic_data = model.prepare_topic_data(
                                    corpus.texts, embeddings=embeddings
                                )
                        end_time = time.time()
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
                            topic_descriptions=topic_data.get_top_words(),
                            top_documents=topic_data.get_top_documents(),
                            top_images=getattr(topic_data, "top_images", None),
                            runtime_s=end_time - start_time,
                            results=res,
                        )
                    except Exception as e:
                        yield BenchmarkEntry.error(
                            dataset=dataset_name,
                            seed=seed,
                            model=model_name,
                            error_message=str(e),
                            n_topics=n_components,
                        )
