import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datasets import load_dataset
from plotly.subplots import make_subplots


def positive(a):
    return np.quantile(a, 0.975) - np.mean(a)


def negative(a):
    return np.mean(a) - np.quantile(a, 0.025)


ds = load_dataset("kardosdrur/topic-benchmark-results", split="train")

encoders = [
    "average_word_embeddings_glove.6B.300d",
    "all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
    "intfloat/e5-large-v2",
]
data = ds.to_pandas()
data = data[data["encoder"].isin(encoders)]
data["model"] = data["model"].replace("S³", "S³_axial")
data["model"] = data["model"].str.replace("S³", "s3")
data = data[data["encoder"] != "average_word_embeddings_glove.6B.300d"]
data = data[data["seed"] == 42]
data = data.drop(columns=["runtime_s", "topic_descriptions"])

metrics = [
    "diversity",
    "wec_in",
    "wec_ex",
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

models = [
    "NMF",
    "LDA",
    "CombinedTM",
    "ZeroShotTM",
    "ECRTM",
    "BERTopic",
    "FASTopic",
    "Top2Vec",
    "s3_axial",
    "s3_angular",
    "s3_combined",
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
        formatted_model = model
        if model.startswith("s3"):
            *_, model_type = model.split("_")
            model_type = model_type[:3]
            formatted_model = f"$S^3_{{\\text{{{model_type}}}}}$"
        fig.append_trace(
            go.Scatter(
                x=[formatted_model],
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
# fig = fig.update_xaxes(tickangle=45, showgrid=False)
fig = fig.update_xaxes(tickangle=90)
fig = fig.update_layout(
    template="plotly_white",
    width=1000,
    height=300,
    showlegend=False,
    margin=dict(l=5, r=5, t=25, b=5),
)
fig = fig.update_layout(
    font=dict(family="Times News Roman", size=16, color="black")
)
fig.show()
fig.write_image("figures/effect_of_preprocessing.png", scale=3)
