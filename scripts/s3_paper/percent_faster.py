from itertools import islice

import numpy as np
import pandas as pd
import plotly.express as px
from datasets import load_dataset


def format_model_name(model: str) -> str:
    if model == "S³":
        model = "S³_axial"
    if model.startswith("S³"):
        _, method = model.split("_")
        model = f"s3_{method}"
    return model


ds = load_dataset("kardosdrur/topic-benchmark-results", split="train")
models = [
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
encoders = [
    "average_word_embeddings_glove.6B.300d",
    "all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
    "intfloat/e5-large-v2",
]
data = ds.to_pandas()
data = data[data["encoder"].isin(encoders)]
data["model"] = data["model"].map(format_model_name)
data = data[data["model"].isin(models)]
data["encoder"] = data["encoder"].str.replace("_", "\\_")
data = data[data["seed"] == 42]

s3_models = ["s3_combined", "s3_axial", "s3_angular"]
runtime_x = []
for index, group in data.groupby(["dataset", "encoder", "n_topics"])[
    ["model", "runtime_s"]
]:
    group = group.set_index("model")["runtime_s"].to_dict()
    s3_runtimes = [group.pop(s3_model) for s3_model in s3_models]
    s3_runtime = np.mean(s3_runtimes)
    for other_runtime in group.values():
        runtime_x.append(other_runtime / s3_runtime)

np.median(runtime_x)

fig = px.violin(
    runtime_x,
    box=True,
    points="all",
    width=400,
    height=400,
    template="plotly_white",
)
fig = fig.update_xaxes(title="")
fig = fig.update_yaxes(title="X Slower than $S^3$")
fig.show()

(index, group), *_ = data.groupby(["dataset", "encoder", "n_topics"])[
    ["model", "runtime_s"]
]
group = group.copy().set_index("model")["runtime_s"].to_dict()

group
