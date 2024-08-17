import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def positive(a):
    return np.quantile(a, 0.975) - np.mean(a)


def negative(a):
    return np.mean(a) - np.quantile(a, 0.025)


results_folder = Path("results/")
files = results_folder.glob("*.jsonl")
entries = []
for result_file in files:
    encoder_name = Path(result_file).stem.replace("__", "/")
    with open(result_file) as in_file:
        # Allows for comments if we want to exclude models.
        for line in in_file:
            if line.startswith("#"):
                continue
            entry = json.loads(line)
            entry["encoder"] = encoder_name
            results = entry.pop("results")
            entry = {**entry, **results}
            if "error_message" not in entry:
                entries.append(entry)

data = pd.DataFrame.from_records(entries)
data = data.drop(columns=["runtime_s", "topic_descriptions"])
data = data.rename(
    columns={
        "Diversity": "$\\text{Diversity}$",
        "Word Embedding Coherence": "$\\text{WEC}_{ex}$",
        "IWEC": "$\\text{WEC}_{in}$",
        "NPMI Coherence": "c-npmi",
    }
)
metrics = [
    "$\\text{Diversity}$",
    "$\\text{WEC}_{in}$",
    "$\\text{WEC}_{ex}$",
]

raw = data[data["dataset"] == "20 Newsgroups Raw"].drop(columns=["dataset"])
preprocessed = data[data["dataset"] == "20 Newsgroups Preprocessed"].drop(
    columns=["dataset"]
)
joint = raw.merge(
    preprocessed,
    on=["n_topics", "encoder", "model"],
    suffixes=["_raw", "_preprocessed"],
)
for metric in metrics:
    joint[f"{metric}_diff"] = (
        joint[f"{metric}_raw"] - joint[f"{metric}_preprocessed"]
    )
joint = joint.set_index(["model"])
joint = joint[[f"{metric}_diff" for metric in metrics]]

Path("figures").mkdir(exist_ok=True)

models = [
    "NMF",
    "LDA",
    "CombinedTM",
    "ZeroShotTM",
    "BERTopic",
    "FASTopic",
    "Top2Vec",
    "S³",
]
summary = joint.groupby("model").agg(["mean", negative, positive])
fig = make_subplots(rows=1, cols=3, subplot_titles=metrics)
for col, metric in enumerate(metrics):
    metric_data = summary[f"{metric}_diff"]
    for model in models:
        model_data = metric_data.loc[model]
        if model_data["mean"] - model_data["negative"] > 0:
            color = "#1670cb"
        elif model_data["mean"] + model_data["positive"] < 0:
            color = "#A5243D"
        else:
            color = "black"
        fig.append_trace(
            go.Scatter(
                x=[model],
                y=[model_data["mean"]],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[model_data["positive"]],
                    arrayminus=[model_data["negative"]],
                    width=0,
                ),
                marker=dict(color=color, size=9),
            ),
            col=col + 1,
            row=1,
        )
fig = fig.add_hline(y=0, line_width=1.5)
fig = fig.update_yaxes(
    range=[-0.7, 0.7],
    showgrid=False,
    # title="$\Delta_{\\text{natural} - \\text{preprocessed}}$",
)
fig = fig.update_xaxes(tickangle=45, showgrid=False)
fig = fig.update_layout(
    template="plotly_white",
    width=1000,
    height=420,
    showlegend=False,
    margin=dict(l=5, r=5, t=25, b=5),
)
fig = fig.update_layout(
    font=dict(family="Times News Roman", size=16, color="black")
)
fig.show()
fig.write_image("figures/effect_of_preprocessing.png", scale=2)
