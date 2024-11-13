from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datasets import load_dataset
from scipy.stats import bootstrap


def bootstrap_interval(data) -> tuple[float, float, float]:
    res = bootstrap([data], np.mean)
    return (
        np.mean(data),
        res.confidence_interval.low,
        res.confidence_interval.high,
    )


def bootstrap_scatter(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    groups: str,
    color_mapping: Optional[dict[str, str]] = None,
) -> go.Figure:
    fig = go.Figure()
    for group_name, group_data in data.groupby(groups):
        x, x_low, x_high = bootstrap_interval(group_data[x_col])
        y, y_low, y_high = bootstrap_interval(group_data[y_col])
        marker = dict()
        if color_mapping is not None and group_name in color_mapping:
            marker["color"] = color_mapping[group_name]
        fig.add_trace(
            go.Scatter(
                name=group_name,
                text=[group_name],
                x=[x],
                y=[y],
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[x_high - x],
                    arrayminus=[x - x_low],
                    thickness=2,
                    width=4,
                ),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[y_high - y],
                    arrayminus=[y - y_low],
                    thickness=2,
                    width=3,
                ),
                showlegend=True,
                mode="markers",
                marker=marker,
            )
        )
    fig = fig.update_layout(template="plotly_white")
    fig = fig.update_traces(marker=dict(size=14))
    return fig


ds = load_dataset("kardosdrur/topic-benchmark-results", split="train")

model_name_mapping = {
    "s3_axial": "$S^3_{ax}$",
    "s3_angular": "$S^3_{ang}$",
    "s3_combined": "$S^3_{com}$",
}
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
data["model"] = data["model"].replace(model_name_mapping)
data = data[data["dataset"] != "20 Newsgroups Preprocessed"]
data = data[data["seed"] == 42]

Path("figures").mkdir(exist_ok=True)

color_mapping = {
    "$S^3_{ax}$": "#125ca5",
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
models = list(color_mapping.keys())

fig = bootstrap_scatter(
    data[data["model"].isin(models)],
    x_col="coherence",
    y_col="diversity",
    groups="model",
    color_mapping=color_mapping,
)
# fig = fig.update_traces(marker=dict(color="black"))
# fig = fig.update_layout(yaxis_range=[0.35, 1.05], xaxis_range=[0.3, 0.47])
fig = fig.update_traces(
    mode="markers",
    showlegend=True,
    textposition="top right",
    marker=dict(size=12),
)
fig = fig.update_layout(
    font=dict(size=14, family="Times New Roman", color="black"),
    width=500,
    height=375,
    margin=dict(b=5, l=5, t=5, r=5),
    xaxis_title="$\\bar{C}\\text{ - Coherence}$",
    yaxis_title="$d\\text{ - Diversity}$",
)
fig.show()
fig.write_image("figures/coherence_diversity.png", scale=3)

paired_runs = data.groupby(["dataset", "seed", "n_topics", "encoder"])
data["interpretability_rank"] = paired_runs[["interpretability"]].rank(
    ascending=False
)
data["runtime_rank"] = paired_runs[["runtime_s"]].rank(ascending=True)

fig = bootstrap_scatter(
    data[data["model"].isin(models)],
    x_col="interpretability_rank",
    y_col="runtime_rank",
    groups="model",
    color_mapping=color_mapping,
)
fig = fig.update_traces(
    mode="markers",
    showlegend=True,
    textposition="top right",
    marker=dict(size=12),
)
fig = fig.update_layout(
    font=dict(size=14, family="Times New Roman", color="black"),
    width=500,
    height=375,
    margin=dict(b=5, l=5, t=5, r=5),
    xaxis_title="$\\text{Rank}(\\sqrt{d \\cdot \\bar{C}})$",
    yaxis_title="$\\text{Rank(Speed)}$",
)
fig = fig.update_xaxes(autorange="reversed")
fig = fig.update_yaxes(autorange="reversed")
fig.show()
fig.write_image("figures/performance_speed.png", scale=3)
