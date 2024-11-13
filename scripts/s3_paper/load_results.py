import pandas as pd
from datasets import load_dataset

encoders = [
    "average_word_embeddings_glove.6B.300d",
    "all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
    "intfloat/e5-large-v2",
]
models = [
    "S³_axial",
    "S³_angular",
    "S³_combined",
    "Top2Vec",
    "FASTopic",
    "ECRTM",
    "BERTopic",
    "CombinedTM",
    "ZeroShotTM",
    "LDA",
    "NMF",
]
ds = load_dataset("kardosdrur/topic-benchmark-results", split="train")
data = ds.to_pandas()
data = data[data["encoder"].isin(encoders)]
data = data[data["model"].isin(models)]
data = data[data["seed"] == 42]
