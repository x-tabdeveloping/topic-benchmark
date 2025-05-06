import base64
from collections import namedtuple
from io import BytesIO, StringIO
from typing import Any, Optional, Type

import msgspec
import numpy as np
from PIL import Image

EntryID = namedtuple("EntryID", ["dataset", "model", "n_topics", "seed"])


def image_enc_hook(obj: Any) -> Any:
    if isinstance(obj, Image.Image):
        # Convert image to base64 encoding
        buffer = BytesIO()
        obj.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    if isinstance(obj, np.str_):
        return str(obj)
    else:
        # Raise a NotImplementedError for other types
        raise NotImplementedError(
            f"Objects of type {type(obj)} are not supported"
        )


def image_dec_hook(type: Type, obj: Any) -> Any:
    # `type` here is the value of the custom type annotation being decoded.
    if type is Image.Image:
        buffer = BytesIO(base64.b64decode(obj))
        img = Image.open(buffer)
        return img
    else:
        # Raise a NotImplementedError for other types
        raise NotImplementedError(f"Objects of type {type} are not supported")


class BenchmarkEntry(msgspec.Struct):
    dataset: str
    model: str
    n_topics: int
    seed: int
    topic_descriptions: list[list[str]]
    runtime_s: float
    results: dict[str, float]
    top_documents: Optional[list[list[str]]] = None
    top_images: Optional[list[list[Image.Image]]] = None
    error_message: Optional[str] = None

    @property
    def entry_id(self) -> EntryID:
        return EntryID(
            self.dataset,
            self.model,
            self.n_topics,
            self.seed,
        )

    @classmethod
    def error(
        cls,
        dataset: str,
        model: str,
        n_topics: int,
        seed: int,
        error_message: str,
    ):
        return cls(
            dataset=dataset,
            model=model,
            n_topics=n_topics,
            seed=seed,
            topic_descriptions=[],
            top_documents=[],
            top_images=[],
            runtime_s=-1,
            results={},
            error_message=error_message,
        )

    @staticmethod
    def encoder():
        return msgspec.json.Encoder(enc_hook=image_enc_hook)

    @classmethod
    def decoder(cls):
        return msgspec.json.Decoder(cls, dec_hook=image_dec_hook)

    def to_json(self):
        return self.encoder().encode(self).decode("utf-8")

    @classmethod
    def from_json(cls, json_str: str):
        return cls.decoder().decode(json_str.encode("utf-8"))
