import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import bootstrap

METRICS_TO_DISPLAY_NAME = {
    "NPMI Coherence": "C\\textsubscript{NPMI}",
    "Word Embedding Coherence": "WEC\\textsubscript{ex}",
    "Diversity": "Diversity",
    "IWEC": "WEC\\textsubscript{in}",
}

METRICS = [
    # "C\\textsubscript{NPMI}",
    "Diversity",
    "WEC\\textsubscript{in}",
    "WEC\\textsubscript{ex}",
]


MODEL_ORDER = [
    "S³",
    "FASTopic",
    # "KeyNMF",
    # "GMM",
    "Top2Vec",
    "BERTopic",
    "CombinedTM",
    "ZeroShotTM",
    "NMF",
    "LDA",
]

EMBEDDING_ORDER = [
    "average_word_embeddings_glove.6B.300d",
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "intfloat/e5-large-v2",
]

DATASET_ORDER = [
    "20 Newsgroups Raw",
    "BBC News",
    "ArXiv ML Papers",
]

results_folder = Path("results/")
files = results_folder.glob("*.jsonl")
out_path = Path("tables/", "main_table.tex")
out_path.parent.mkdir(exist_ok=True)
encoder_entries = dict()
entries = []
for result_file in files:
    encoder_name = Path(result_file).stem.replace("__", "/")
    with open(result_file) as in_file:
        for line in in_file:
            # Allows for comments if we want to exclude models.
            if line.startswith("#"):
                continue
            entry = json.loads(line)
            entry["encoder"] = encoder_name
            res = entry.pop("results")
            entry = {**entry, **res}
            if entry["model"] in MODEL_ORDER:
                entries.append(entry)

data = pd.DataFrame.from_records(entries)
include = (
    data["dataset"].isin(DATASET_ORDER)
    & data["model"].isin(MODEL_ORDER)
    & data["encoder"].isin(EMBEDDING_ORDER)
)
data = data[include]

data["performance_rank"] = (
    data.groupby(["dataset", "encoder", "n_topics"])[
        ["IWEC", "Diversity", "Word Embedding Coherence"]
    ]
    .rank(ascending=False)
    .mean(axis=1)
)
data["speed_rank"] = data.groupby(["dataset", "encoder", "n_topics"])[
    ["runtime_s"]
].rank(ascending=True)


def bootstrap_rank(ranks) -> tuple[float, float, float]:
    mean_rank = np.mean(ranks)
    res = bootstrap([ranks], np.mean)
    return mean_rank, res.confidence_interval.low, res.confidence_interval.high


MODEL_TO_COLOR = {
    "S³": "#1670CB",
    "FASTopic": "black",
    "CombinedTM": "black",
    "ZeroShotTM": "black",
    "Top2Vec": "black",
    "BERTopic": "black",
    "NMF": "black",
    "LDA": "black",
}
fig = go.Figure()
# fig = fig.add_shape(
#     type="rect",
#     xref="x",
#     yref="y",
#     x0=3.5,
#     x1=1.9,
#     y0=4.5,
#     y1=2.2,
#     fillcolor="rgba(0,0,0,0)",
#     line=dict(color="#1670CB", width=2, dash="dash"),
#     opacity=0.2,
# )
# fig = fig.add_annotation(
#     x=3.5,
#     y=2.0,
#     text="Fast and Performant",
#     showarrow=False,
#     xshift=0,
#     yshift=0,
#     font=dict(color="#1670CB"),
#     xanchor="left",
# )
# fig = fig.add_shape(
#     type="rect",
#     xref="x",
#     yref="y",
#     x0=4.1,
#     x1=3,
#     y0=6.6,
#     y1=5.2,
#     fillcolor="rgba(0,0,0,0)",
#     line=dict(color="black", width=2, dash="dash"),
#     opacity=0.2,
# )
# fig = fig.add_annotation(
#     x=4.15,
#     y=5.0,
#     text="Slow but Performant",
#     showarrow=False,
#     xshift=0,
#     yshift=0,
#     xanchor="left",
# )
for model in MODEL_ORDER:
    model_data = data[data["model"] == model]
    s, s_low, s_high = bootstrap_rank(model_data["speed_rank"])
    p, p_low, p_high = bootstrap_rank(model_data["performance_rank"])
    color = MODEL_TO_COLOR[model]
    fig.add_trace(
        go.Scatter(
            name=model,
            y=[s],
            x=[p],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[s_high - s],
                arrayminus=[s - s_low],
                thickness=2,
                width=3,
            ),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[p_high - p],
                arrayminus=[p - p_low],
                thickness=2,
                width=4,
            ),
            showlegend=False,
            marker=dict(color=color),
        )
    )
    yshift = 16
    xshift = 2
    xanchor = "left"
    if model == "BERTopic":
        yshift = 35
        xanchor = "center"
    if model == "CombinedTM":
        yshift = -10
    fig.add_annotation(
        x=p,
        y=s,
        text=model,
        showarrow=False,
        yshift=yshift,
        xshift=xshift,
        font=dict(size=18, color=color),
        xanchor=xanchor,
    )
fig = fig.update_layout(template="plotly_white", width=500, height=330)
fig = fig.update_traces(marker=dict(size=14))
fig = fig.update_yaxes(autorange="reversed").update_xaxes(autorange="reversed")
fig = fig.update_layout(
    xaxis_title="Average Performance Rank",
    yaxis_title="Average Speed Rank",
    font=dict(family="Times New Roman", size=16, color="black"),
    margin=dict(b=0, t=0, l=0, r=0),
)
fig.show()

fig.write_image("figures/ranks.png", scale=3)
