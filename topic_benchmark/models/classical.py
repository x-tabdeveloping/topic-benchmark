from typing import List, Literal, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, _name_estimators
from turftopic.data import TopicData

from topic_benchmark import model_registry
from topic_benchmark.base import Loader, TopicModel


class TopicPipeline(Pipeline, TopicModel):
    """Scikit-learn compatible topic pipeline."""

    def __init__(
        self,
        steps: List[Tuple[str, BaseEstimator]],
        *,
        memory=None,
        verbose=False,
    ):
        super().__init__(steps, memory=memory, verbose=verbose)

    def prepare_topic_data(
        self,
        corpus: List[str],
        document_representation: Literal["term", "topic"] = "term",
    ) -> TopicData:
        """Prepares topic data"""
        try:
            print("Inferring topical content for documents.")
            document_topic_matrix = np.asarray(self.transform(corpus))
        except (NotFittedError, AttributeError) as e:
            if e is NotFittedError:
                print("Pipeline has not been fitted, fitting.")
            if e is AttributeError:
                print(
                    "Looks like the topic model is transductive. Running fit_transform()"
                )
            document_topic_matrix = np.asarray(self.fit_transform(corpus))
        try:
            components = self.topic_model.components_  # type: ignore
        except AttributeError as e:
            raise ValueError(
                "Topic model does not have components_ attribute."
            ) from e
        document_term_matrix = self.vectorizer.transform(corpus)  # type: ignore
        vocab = self.vectorizer.get_feature_names_out()  # type: ignore
        res = TopicData(
            corpus=corpus,
            document_term_matrix=document_term_matrix,
            document_topic_matrix=document_topic_matrix,
            document_representation=document_term_matrix
            if document_representation == "term"
            else document_topic_matrix,
            vocab=vocab,
            topic_term_matrix=components,
            transform=self.transform,
            topic_names=[],
        )
        return res


def make_topic_pipeline(
    *steps,
    memory=None,
    verbose=False,
):
    """Shorthand for constructing a topic pipeline."""
    return TopicPipeline(
        _name_estimators(steps),
        memory=memory,
        verbose=verbose,
    )


@model_registry.register("NMF")
def load_nmf() -> Loader:
    def _load(n_components: int, encoder, vectorizer: CountVectorizer):
        model = NMF(n_components)
        return make_topic_pipeline(vectorizer, model)

    return _load


@model_registry.register("LDA")
def load_lda() -> Loader:
    def _load(n_components: int, encoder, vectorizer: CountVectorizer):
        model = LatentDirichletAllocation(n_components)
        return make_topic_pipeline(vectorizer, model)

    return _load
