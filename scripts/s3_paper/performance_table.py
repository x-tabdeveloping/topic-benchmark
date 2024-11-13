from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
from datasets import load_dataset


def safe_index(elements: list, elem) -> Optional[int]:
    try:
        return elements.index(elem)
    except ValueError:
        return None


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
data["coherence"] = np.sqrt(data["wec_in"] * data["wec_ex"])
data["interpretability"] = np.sqrt(data["diversity"] * data["coherence"])
data["model"] = data["model"].replace("S³", "S³_axial")
data["model"] = data["model"].str.replace("S³", "s3")
data = data[data["seed"] == 42]
summary = (
    data.groupby(["dataset", "model"])[
        ["diversity", "wec_in", "wec_ex", "interpretability", "coherence"]
    ]
    .mean()
    .reset_index()
)


METRICS = ["diversity", "wec_in", "wec_ex", "interpretability"]
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
lines = []
for dataset, res in summary.groupby("dataset"):
    lines.append("\\midrule")
    lines.append(f"{dataset} & & & &\\\\")
    lines.append("\\midrule")
    res = res[res["model"].isin(MODELS)]
    res = res.set_index("model")
    best_models = {
        metric: list(res[metric].nlargest(2).index) for metric in METRICS
    }
    for model in MODELS:
        if model not in res.index:
            continue
        entry = res.loc[model]
        formatted_metrics = []
        for metric in METRICS:
            metric_result = entry[metric]
            metric_formatted = f"{metric_result:.2f}"
            model_rank = safe_index(best_models[metric], model)
            if model_rank == 0:
                metric_formatted = f"\\textbf{{{metric_formatted}}}"
            if model_rank == 1:
                metric_formatted = f"\\underline{{{metric_formatted}}}"
            formatted_metrics.append(metric_formatted)
        formatted_model = model
        if model.startswith("s3"):
            *_, model_type = model.split("_")
            model_type = model_type[:3]
            formatted_model = f"$S^3_{{\\text{{{model_type}}}}}$"
        lines.append(
            f"{formatted_model} &" + " & ".join(formatted_metrics) + "\\\\"
        )
    if dataset == "ArXiv ML Papers":
        lines.append("\n\n")
print("\n".join(lines))
