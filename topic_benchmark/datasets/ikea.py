from datasets import load_dataset

from topic_benchmark.datasets.dataset import Dataset
from topic_benchmark.registries import dataset_registry


@dataset_registry.register("IKEA")
def load_ikea():
    ds = load_dataset("nbadrinath/ikea_dataset_5.0", split="train[:500]")
    images = ds["image"]
    texts = ds["desc"]
    nans = []
    for image, text in zip(images, texts):
        if (image is None) or (text is None):
            nans.append(True)
        else:
            nans.append(False)
    images = [image for image, is_nan in zip(images, nans) if not is_nan]
    texts = [text for text, is_nan in zip(texts, nans) if not is_nan]
    return Dataset(list(texts), images=list(images))
