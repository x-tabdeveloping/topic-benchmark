from typing import Optional

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer
from turftopic import SemanticSignalSeparation
from turftopic.data import TopicData
from turftopic.multimodal import ImageRepr, MultimodalEmbeddings

from topic_benchmark.base import Loader
from topic_benchmark.registries import model_registry


class DoubleS3(SemanticSignalSeparation):
    def prepare_topic_data(
        self,
        corpus: list[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> TopicData:
        if embeddings is None:
            embeddings = self.encode_documents(corpus)
        try:
            document_topic_matrix = self.transform(
                corpus, embeddings=embeddings
            )
        except (AttributeError, NotFittedError):
            document_topic_matrix = self.fit_transform(
                corpus, embeddings=embeddings
            )
        # adding the negative sides to the document_topic and topic_term matrices
        document_topic_matrix = np.concatentate(
            (document_topic_matrix, -document_topic_matrix), axis=1
        )
        components = np.concatentate(
            (self.components_, -self.components_), axis=0
        )
        dtm = self.vectorizer.transform(corpus)  # type: ignore
        classes = list(range(components.shape[0]))
        res = TopicData(
            corpus=corpus,
            document_term_matrix=dtm,
            vocab=self.get_vocab(),
            document_topic_matrix=document_topic_matrix,
            document_representation=embeddings,
            topic_term_matrix=components,  # type: ignore
            transform=getattr(self, "transform", None),
            topic_names=[str(i) for i in classes],
            classes=classes,
            has_negative_side=self.has_negative_side,
            hierarchy=getattr(self, "hierarchy", None),
        )
        return res

    def prepare_multimodal_topic_data(
        self,
        corpus: list[str],
        images: list[ImageRepr],
        embeddings: Optional[MultimodalEmbeddings] = None,
    ) -> TopicData:
        if embeddings is None:
            embeddings = self.encode_multimodal(corpus, images)
        document_topic_matrix = self.fit_transform_multimodal(
            corpus, images=images, embeddings=embeddings
        )
        dtm = self.vectorizer.transform(corpus)  # type: ignore
        document_topic_matrix = np.concatentate(
            (document_topic_matrix, -document_topic_matrix), axis=1
        )
        components = np.concatentate(
            (self.components_, -self.components_), axis=0
        )
        classes = list(range(components.shape[0]))
        top_images = [*self.top_images, *self.negative_images]
        res = TopicData(
            corpus=corpus,
            document_term_matrix=dtm,
            vocab=self.get_vocab(),
            document_topic_matrix=document_topic_matrix,
            document_representation=embeddings["document_embeddings"],
            topic_term_matrix=components,  # type: ignore
            transform=getattr(self, "transform", None),
            topic_names=self.topic_names,
            classes=classes,
            has_negative_side=self.has_negative_side,
            hierarchy=getattr(self, "hierarchy", None),
            images=images,
            top_images=top_images,
            negative_images=None,
        )
        return res


@model_registry.register("S3")
def load_double_sided_s3(encoder, vectorizer: CountVectorizer) -> Loader:
    """This is diferent from the original S3 implementation,
    in that it returns both sides of the topic instead of just one.
    I made this the default option, since it more accurately reflects S3's behaviour in metrics.
    """

    def _load(n_components: int, seed: int):
        return DoubleS3(
            n_components,
            encoder=encoder,
            vectorizer=vectorizer,
            random_state=seed,
            feature_importance="combined",
        )

    return _load
