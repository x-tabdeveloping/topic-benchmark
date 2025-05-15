import warnings
from typing import Optional

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from turftopic.data import TopicData

from topic_benchmark.base import Metric
from topic_benchmark.encoders.clip import CLIPModelWrapper
from topic_benchmark.registries import metric_registry

DEFAULT_ENCODER = "openai/clip-vit-base-patch32"


def encode_descriptives(
    data: TopicData, top_k: int, encoder
) -> tuple[np.ndarray, np.ndarray]:
    """Encodes descriptive words, documents and images."""
    top_words = data.get_top_words(top_k=top_k)
    top_documents = data.get_top_documents(top_k=top_k)
    embeddings = []
    labels = []
    for name, words, docs in zip(data.topic_names, top_words, top_documents):
        if name.startswith("-1"):
            # Skipping outlier topic
            continue
        embeddings.extend(encoder.get_text_embeddings(words).tolist())
        labels.extend([name] * len(words))
        embeddings.extend(encoder.get_text_embeddings(docs).tolist())
        labels.extend([name] * len(docs))
    try:
        top_images = data.get_top_images(top_k=top_k)
        for name, images in zip(data.topic_names, top_images):
            if name.startswith("-1"):
                # Skipping outlier topic
                continue
            embeddings.extend(encoder.get_image_embeddings(images).tolist())
            labels.extend([name] * len(images))
    except Exception:
        warnings.warn("No images, proceeding without them.")
        pass
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    return embeddings, labels


@metric_registry.register("silhouette")
def load_silhouette(top_k: int = 10) -> Metric:
    encoder = CLIPModelWrapper("openai/clip-vit-base-patch32")

    def score(data: TopicData, dataset_name: Optional[str]):
        embeddings, labels = encode_descriptives(data, top_k, encoder)
        res = silhouette_score(embeddings, labels, metric="cosine")
        return float(res)

    return score


@metric_registry.register("coverage")
def load_coverage(top_k: int = 10) -> Metric:
    encoder = CLIPModelWrapper("openai/clip-vit-base-patch32")

    # Cache for dataset embeddings
    embedding_cache: dict[str, np.ndarray] = {}

    def score(data: TopicData, dataset_name: Optional[str]):
        if dataset_name not in embedding_cache:
            if getattr(data, "images", None) is None:
                warnings.warn(
                    "Corpus doesn't have images, encoding text only."
                )
                embedding_cache[dataset_name] = encoder.get_text_embeddings(
                    data.corpus
                )
            else:
                embedding_cache[dataset_name] = encoder.get_fused_embeddings(
                    data.corpus, images=data.images
                )
        doc_emb = embedding_cache[dataset_name]
        desc_emb, _ = encode_descriptives(data, top_k, encoder)
        sim = cosine_similarity(doc_emb, desc_emb)
        # Selecting the maximally similar description item to a document
        max_sim = np.max(sim, axis=1)
        return np.mean(max_sim)

    return score
