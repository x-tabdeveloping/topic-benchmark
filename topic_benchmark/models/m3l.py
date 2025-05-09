from typing import Optional, Union

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from turftopic.multimodal import (ImageRepr, MultimodalEmbeddings,
                                  MultimodalModel)

from topic_benchmark.base import Loader
from topic_benchmark.defaults import default_vectorizer
from topic_benchmark.models._m3l import M3LDataset, MultimodalContrastiveTM
from topic_benchmark.registries import model_registry


class M3LWrapper(MultimodalModel):
    """Implementation of M3L with the Turftopic API."""

    def __init__(
        self,
        n_components: int,
        encoder,
        vectorizer: Optional[CountVectorizer] = None,
        random_state: Optional[int] = None,
        batch_size: Optional[int] = 200,
        n_epochs: int = 64,
        learning_rate: float = 0.002,
        device: str = "cpu",
    ):
        self.n_components = n_components
        self.encoder_ = encoder
        if random_state is not None:
            self.random_state = random_state
        else:
            self.random_state = 0
        if vectorizer is None:
            self.vectorizer = default_vectorizer()
        else:
            self.vectorizer = vectorizer
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.batch_size = batch_size

    def get_vocab(self) -> np.ndarray:
        """Get vocabulary of the model.

        Returns
        -------
        ndarray of shape (n_vocab)
            All terms in the vocabulary.
        """
        return self.vectorizer.get_feature_names_out()

    @property
    def components_(self) -> np.ndarray:
        return self.model.components_

    @property
    def has_negative_side(self):
        return False

    def fit_transform_multimodal(
        self,
        raw_documents: list[str],
        images: list[ImageRepr],
        y=None,
        embeddings: Optional[MultimodalEmbeddings] = None,
    ) -> np.ndarray:
        torch.manual_seed(self.random_state)
        if embeddings is None:
            embeddings = self.encode_multimodal(raw_documents, images)
        dtm = self.vectorizer.fit_transform(raw_documents)
        # Don't ask me why you have to wrap some things in lists but not others
        # This is literally the worst piece of research code I've had to work with
        training_dataset = M3LDataset(
            dtm=dtm,
            embeddings=embeddings["text_embeddings"],
            image_embeddings=embeddings["image_embeddings"],
        )
        self.model = MultimodalContrastiveTM(
            bow_size=dtm.shape[1],
            contextual_sizes=(
                embeddings["text_embeddings"].shape[1],
                embeddings["image_embeddings"].shape[1],
            ),
            n_components=self.n_components,
            batch_size=self.batch_size,
            lr=self.learning_rate,
        )
        self.model.fit(training_dataset)
        self.document_topic_matrix = np.array(
            self.model.get_doc_topic_distribution(
                M3LDataset(dtm, embeddings=embeddings["document_embeddings"]),
            )
        )
        self.image_topic_matrix = np.array(
            self.model.get_doc_topic_distribution(
                M3LDataset(
                    dtm, image_embeddings=embeddings["image_embeddings"]
                ),
            )
        )
        self.top_images = self.collect_top_images(
            images, self.image_topic_matrix
        )
        self.topic_names = [str(i) for i in range(self.n_components)]
        return self.document_topic_matrix


@model_registry.register("M3L")
def load_m3l(encoder, vectorizer: CountVectorizer) -> Loader:
    def _load(n_components: int, seed: int):
        return M3LWrapper(
            n_components,
            encoder=encoder,
            vectorizer=vectorizer,
            random_state=seed,
        )

    return _load
