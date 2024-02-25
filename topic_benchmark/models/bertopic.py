from functools import partial
from typing import Optional

import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import label_binarize
from turftopic.data import TopicData

from topic_benchmark.base import Loader, TopicModel
from topic_benchmark.registries import model_registry


class BERTopicWrapper(TopicModel):
    """Wrapper for BERTopic models to be used in topicwizard."""

    def __init__(self, model):
        self.model = model

    def prepare_topic_data(
        self, corpus: list[str], embeddings: Optional[np.ndarray] = None
    ) -> TopicData:
        from bertopic.backend._utils import select_backend

        if embeddings is None:
            self.model.embedding_model = select_backend(
                self.model.embedding_model, language=self.model.language
            )
            embeddings = self.model._extract_embeddings(
                corpus,
                method="document",
            )
        if self.model.c_tf_idf_ is None:
            topic_labels, _ = self.model.fit_transform(
                corpus, embeddings=embeddings
            )
        else:
            topic_labels, _ = self.model.transform(
                corpus, embeddings=embeddings
            )
        document_topic_matrix = label_binarize(
            topic_labels, classes=self.model.topics_
        )
        document_term_matrix = self.model.vectorizer_model.transform(corpus)
        vocab = self.model.vectorizer_model.get_feature_names_out()
        if self.model.topic_labels_:
            topic_names = [
                self.model.topic_labels_[topic] for topic in self.model.topics_
            ]
        else:
            topic_names = self.model.generate_topic_labels(nr_words=3)

        def transform(corpus: list[str]):
            topic_labels, _ = self.model.transform(
                corpus, embeddings=embeddings
            )
            return label_binarize(topic_labels, classes=self.model.topics_)

        return TopicData(
            corpus=corpus,
            vocab=vocab,
            document_term_matrix=document_term_matrix,
            document_topic_matrix=np.asarray(document_topic_matrix),
            topic_term_matrix=self.model.c_tf_idf_.toarray(),
            document_representation=embeddings,  # type: ignore
            transform=transform,
            topic_names=topic_names,
        )


@model_registry.register("BERTopic")
def load_bertopic(encoder, vectorizer: CountVectorizer) -> Loader:
    def _load(n_components: int):
        model = BERTopic(
            embedding_model=encoder,
            vectorizer_model=vectorizer,
            nr_topics=n_components,
        )
        return BERTopicWrapper(model)

    return _load
