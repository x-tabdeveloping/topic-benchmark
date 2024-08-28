import math
from typing import Optional, Union

import numpy as np
import torch
from tqdm import trange
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from turftopic.data import TopicData

from topic_benchmark.models._ecrtm import ECRTMModule
from topic_benchmark.defaults import default_vectorizer
from topic_benchmark.registries import model_registry
from topic_benchmark.base import Loader


class ECRTM:
    """
    Implementation of ECRTM with a Turftopic API.
    The implementation is based on the [TopMost](https://github.com/BobXWu/TopMost/blob/main/topmost/models/basic/ECRTM/ECRTM.py),
    """

    def __init__(
        self,
        n_components: int,
        vectorizer: Optional[CountVectorizer] = None,
        random_state: Optional[int] = None,
        batch_size: Optional[int] = 200,
        n_epochs: int = 200,
        learning_rate: float = 0.002,
        device: str = "cpu",
    ):
        self.n_components = n_components
        self.random_state = random_state
        if vectorizer is None:
            self.vectorizer = default_vectorizer()
        else:
            self.vectorizer = vectorizer
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.batch_size = batch_size

    def make_optimizer(self, learning_rate: float):
        args_dict = {
            "params": self.model.parameters(),
            "lr": learning_rate,
        }
        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def _train_model(self, document_term_matrix):
        torch.manual_seed(self.random_state)
        self.model = ECRTMModule(vocab_size=document_term_matrix.shape[1], num_topics=self.n_components)
        self.model = self.model.to(self.device)
        optimizer = self.make_optimizer(self.learning_rate)
        self.model.training=True
        if self.batch_size is None:
            batch_size = document_term_matrix.shape[0]
        else:
            batch_size = self.batch_size
        num_batches = int(
            math.ceil(document_term_matrix.shape[0] / batch_size)
        )
        for epoch in trange(self.n_epochs,  desc="Training epochs"):
            running_loss = 0
            for i in range(num_batches):
                batch_bow = np.atleast_2d(
                    document_term_matrix[
                        i * batch_size : (i + 1) * batch_size, :
                    ].toarray()
                )
                # Skipping batches that are smaller than 2
                if batch_bow.shape[0] < 2:
                    continue
                batch_bow = torch.tensor(batch_bow).float().to(self.device)
                rst_dict = self.model(batch_bow)
                batch_loss = rst_dict["loss"]
                running_loss += batch_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
        self.components_ = self.model.get_beta().detach().numpy()
        self.model.training=False

    def fit_transform(self, raw_documents, y=None) -> np.ndarray:
        document_term_matrix = self.vectorizer.fit_transform(raw_documents)
        self._train_model(document_term_matrix)
        document_topic_matrix = self.transform(raw_documents)
        return document_topic_matrix

    def transform(self, raw_documents) -> np.ndarray:
        document_term_matrix = self.vectorizer.transform(raw_documents)
        if self.batch_size is None:
            batch_size = document_term_matrix.shape[0]
        else:
            batch_size = self.batch_size
        num_batches = int(
            math.ceil(document_term_matrix.shape[0] / batch_size)
        )
        batch_thetas = []
        with torch.no_grad():
            for i in range(num_batches):
                batch_bow = np.atleast_2d(
                    document_term_matrix[
                        i * batch_size : (i + 1) * batch_size, :
                    ].toarray()
                )
                batch_theta = self.model.get_theta(
                    torch.as_tensor(batch_bow).float(),
                )
                batch_theta = batch_theta.detach().cpu().numpy()
                batch_thetas.append(batch_theta)
        return np.concatenate(batch_thetas, axis=0)

    def get_vocab(self) -> np.ndarray:
        return self.vectorizer.get_feature_names_out()

    def prepare_topic_data(
        self,
        corpus: list[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> TopicData:
        document_topic_matrix = self.fit_transform(corpus)
        dtm = self.vectorizer.transform(corpus)  # type: ignore
        res: TopicData = {
            "corpus": corpus,
            "document_term_matrix": dtm,
            "vocab": self.get_vocab(),
            "document_topic_matrix": document_topic_matrix,
            "document_representation": document_topic_matrix,
            "topic_term_matrix": self.components_,  # type: ignore
            "transform": getattr(self, "transform", None),
            "topic_names": [],
        }
        return res

@model_registry.register("ECRTM")
def load_ecrtm(encoder, vectorizer: CountVectorizer) -> Loader:
    def _load(n_components: int, seed: int):
        return ECRTM(
            n_components,
            vectorizer=vectorizer,
            random_state=seed,
        )

    return _load

