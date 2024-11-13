from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
from datasets import load_dataset


def format_models(model):
    if model.startswith("s3"):
        *_, model_type = model.split("_")
        model_type = model_type[:3]
        return f"$S^3_{{\\text{{{model_type}}}}}$"
    if model in ["ZeroShotTM", "CombinedTM"]:
        model_type = model.removesuffix("TM").lower()
        return f"$\\text{{CTM}}_{{\\text{{{model_type}}}}}$"
    return f"$\\text{{{model}}}$"


MODELS = [
    "s3_axial",
    "s3_angular",
    "s3_combined",
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
encoders = [
    "average_word_embeddings_glove.6B.300d",
    "all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
    "intfloat/e5-large-v2",
]
data = ds.to_pandas()
data = data[data["encoder"].isin(encoders)]
data = data[data["encoder"] != "average_word_embeddings_glove.6B.300d"]
data["model"] = data["model"].replace("S³", "S³_axial")
data["model"] = data["model"].str.replace("S³", "s3")
data = data[data["model"].isin(MODELS)]
data = data[data["seed"] == 42]

summary = data.groupby(["model", "dataset"])["c_npmi"].mean().reset_index()
summary = summary.pivot(index="model", columns="dataset", values="c_npmi")
summary = summary.loc[MODELS]
summary.index = list(map(format_models, summary.index))
summary = summary.map(lambda val: f"{val:.2f}")

print(summary.to_latex())
