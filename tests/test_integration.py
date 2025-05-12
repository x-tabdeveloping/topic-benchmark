from pathlib import Path

from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from topic_benchmark.cli import run_cli
from topic_benchmark.datasets.dataset import Dataset
from topic_benchmark.registries import dataset_registry


@dataset_registry.register("dummy_ikea")
def load_ikea():
    ds = load_dataset("nbadrinath/ikea_dataset_5.0", split="test[:64]")
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


@dataset_registry.register("dummy_20NG")
def load_newsgroups_raw() -> list[str]:
    ds = fetch_20newsgroups(subset="all")
    corpus = list(ds.data)[:32]
    return Dataset(corpus)


OUT_FOLDER = Path(__file__).parent.joinpath("__test_results")

METRICS = ["wec_in", "coverage", "diversity", "silhouette"]


def test_monomodal():
    run_cli(
        out_dir=OUT_FOLDER,
        encoders=["all-MiniLM-L6-v2"],
        models=["S3"],
        datasets=["dummy_20NG"],
        seeds=[42],
        multimodal=False,
        strict=True,
        metrics=METRICS,
    )


def test_multimodal():
    run_cli(
        out_dir=OUT_FOLDER,
        encoders=["openai/clip-vit-base-patch32"],
        models=["GMM"],
        datasets=["dummy_ikea"],
        seeds=[42],
        multimodal=True,
        strict=True,
        metrics=METRICS,
    )
