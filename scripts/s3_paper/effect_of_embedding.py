import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datasets import load_dataset
from plotly.subplots import make_subplots


def format_models(model):
    if model.startswith("s3"):
        *_, model_type = model.split("_")
        model_type = model_type[:3]
        return f"$S^3_{{{model_type}}}$"
    return model


def positive(a):
    return np.quantile(a, 0.975) - np.mean(a)


def negative(a):
    return np.mean(a) - np.quantile(a, 0.025)


ds = load_dataset("kardosdrur/topic-benchmark-results", split="train")

models = [
    "BERTopic",
    "CombinedTM",
    "ZeroShotTM",
    "FASTopic",
    "Top2Vec",
    "s3_axial",
    "s3_angular",
    "s3_combined",
]
encoders = [
    "GloVe",
    "all-MiniLM-L6",
    "all-mpnet-base",
    "e5-large",
]
data = ds.to_pandas()
data["model"] = data["model"].replace("S³", "S³_axial")
data["model"] = data["model"].str.replace("S³", "s3")
data["coherence"] = np.sqrt(data["wec_in"] * data["wec_ex"])
data["interpretability"] = np.sqrt(data["diversity"] * data["coherence"])
# data = data[data["encoder"] != "average_word_embeddings_glove.6B.300d"]
data = data[data["seed"] == 42]
data = data.drop(columns=["runtime_s", "topic_descriptions"])
data = data[data["model"].isin(models)]
data["encoder"] = data["encoder"].replace(
    {
        "average_word_embeddings_glove.6B.300d": "GloVe",
        "intfloat/e5-large-v2": "e5-large-v2",
    }
)
data["encoder"] = data["encoder"].map(lambda m: m.removesuffix("-v2"))
data = data[data["encoder"].isin(encoders)]
data["model"] = data["model"].map(format_models)
model_order = list(map(format_models, models))

summary = (
    data.groupby(["model", "encoder"])["interpretability"].mean().reset_index()
)

summary

color_mapping = {
    "$S^3_{axi}$": "#125ca5",
    "$S^3_{ang}$": "#4799EB",
    "$S^3_{com}$": "#1670CB",
    "Top2Vec": "#44BBA4",
    "FASTopic": "#E94F37",
    "ECRTM": "#B82A14",
    "BERTopic": "#2B7869",
    "CombinedTM": "#F3B700",
    "ZeroShotTM": "#A37A00",
    "LDA": "#C09BD8",
    "NMF": "#8F4FBA",
}
fig = px.line(
    summary,
    x="encoder",
    y="interpretability",
    color="model",
    template="plotly_white",
    category_orders={
        "model": model_order,
        "encoder": encoders,
    },
    labels={
        "encoder": "Embedding Model",
        "interpretability": "$\\sqrt{d \\cdot \\bar{C}}$",
    },
    color_discrete_map=color_mapping,
    width=475,
    height=250,
)
fig = fig.update_layout(legend_title_text="", legend_traceorder="reversed")
fig = fig.update_layout(
    font=dict(family="Times New Roman", color="black", size=14)
)
fig = fig.update_xaxes(tickangle=0)
fig = fig.update_layout(margin=dict(b=0, l=0, r=0, t=0))
fig = fig.update_traces(line=dict(width=2.5))
fig = fig.update_layout(xaxis_title="")
fig.show()
fig.write_image("figures/effect_of_embedding_model.png", scale=3)
