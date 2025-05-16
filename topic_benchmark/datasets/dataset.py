from typing import Optional, Union

import numpy as np
from turftopic.multimodal import ImageRepr, MultimodalEmbeddings


class Dataset:
    """Dataset class that can also behave like a list of strings"""

    def __init__(
        self,
        texts: list[str],
        images: Optional[list[ImageRepr]] = None,
    ):
        self.texts = texts
        self.images = images

    def __getitem__(self, item):
        return self.texts[item]

    def __len__(self):
        return len(self.texts)

    @property
    def is_multimodal(self):
        return self.images is not None
