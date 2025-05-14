import os
from pathlib import Path

os.chdir(Path(__file__).parent.parent)

from sklearn.feature_extraction.text import CountVectorizer

from topic_benchmark.cli import run_cli
from topic_benchmark.datasets.dataset import Dataset
from topic_benchmark.registries import dataset_registry

OUT_FOLDER = "multimodal_results/"
METRICS = [
    "wec_in",
    "wec_ex",
    "coverage",
    "diversity",
    "silhouette",
    "stop_freq",
]
DATASETS = [
    "VisualNews",
    "InfoSeek",
    "Oven",
    "Edis",
    "WebQA",
    "Fashion200k",
    "MSCOCO",
]
MULTIMODAL_MODELS = [
    "S3",
    "KeyNMF",
    "GMM",
    "BERTopic",
    "Top2Vec",
    "MultimodalContrastTM",
    "MultimodalZeroShotTM",
]
ENCODERS = [
    "openai/clip-vit-base-patch32",
    "royokong/e5-v",
]
SEEDS = [42]


def main() -> None:
    run_cli(
        out_dir=OUT_FOLDER,
        encoders=ENCODERS,
        models=MULTIMODAL_MODELS,
        datasets=DATASETS,
        seeds=SEEDS,
        multimodal=True,
        strict=False,
        metrics=METRICS,
    )


if __name__ == "__main__":
    main()
