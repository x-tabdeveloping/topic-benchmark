from abc import abstractmethod
from typing import Any, Callable, Protocol

from turftopic.data import TopicData


class TopicModel(Protocol):
    @abstractmethod
    def prepare_topic_data(
        self,
        corpus: list[str],
        *args: Any,
        **kwargs: Any,
    ) -> TopicData:
        ...


Loader = Callable[[int], TopicModel]

Metric = Callable[[TopicData], float]
