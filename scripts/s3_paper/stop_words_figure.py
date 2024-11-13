import itertools
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from datasets import load_dataset
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

STOPS = set(ENGLISH_STOP_WORDS)


def stop_word_rel_freq(topic_descriptions: list[list[str]]) -> float:
    words = list(itertools.chain.from_iterable(topic_descriptions))
    total = len(words)
    n_stop = 0
    for word in words:
        if word in STOPS:
            n_stop += 1
    return n_stop / total


def nonalpha_rel_freq(topic_descriptions: list[list[str]]) -> float:
    words = list(itertools.chain.from_iterable(topic_descriptions))
    total = len(words)
    n_nonalpha = 0
    for word in words:
        if not all([c.isalpha() for c in word]):
            n_nonalpha += 1
    return n_nonalpha / total


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
data = data[data["seed"] == 42]
data = data[data["dataset"] != "20 Newsgroups Preprocessed"]
data = data[data["encoder"] != "average_word_embeddings_glove.6B.300d"]

data["stop_freq"] = data["topic_descriptions"].map(stop_word_rel_freq)
data["nonalpha_freq"] = data["topic_descriptions"].map(nonalpha_rel_freq)


def s3_map_names(name: str) -> str:
    if name.startswith("s3"):
        *_, model_type = name.split("_")
        model_type = model_type[:3]
        formatted_model = f"$S^3_{{\\text{{{model_type}}}}}$"
        return formatted_model
    return name


Path("figures").mkdir(exist_ok=True)
for column in ["stop_freq", "nonalpha_freq"]:
    models = [
        "NMF",
        "LDA",
        "BERTopic",
        "CombinedTM",
        "ZeroShotTM",
        "FASTopic",
        "ECRTM",
        "Top2Vec",
        "s3_axial",
        "s3_angular",
        "s3_combined",
    ]
    models = list(map(s3_map_names, models))
    data["model"] = data["model"].map(s3_map_names)
    data = data[data["model"].isin(models)]
    fig = go.Figure()
    fig = fig.add_box(
        y=data["model"],
        x=data[column],
        fillcolor="white",
        marker=dict(color="black"),
    )
    fig = fig.update_traces(orientation="h")
    fig = fig.update_layout(width=350, height=275, template="plotly_white")
    fig = fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
    fig = fig.update_yaxes(categoryorder="array", categoryarray=models[::-1])
    if column == "stop_freq":
        xaxis_title = "Proportion of Stop Words"
    else:
        xaxis_title = "Freq. of Nonalphabetical Words"
    fig = fig.update_xaxes(title=xaxis_title)
    fig = fig.update_layout(
        font=dict(family="Times New Roman", size=14, color="black")
    )
    fig.write_image(f"figures/{column}.png", scale=2)
    fig.show()
