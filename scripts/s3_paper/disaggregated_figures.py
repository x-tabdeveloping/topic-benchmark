from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset


CATEGORY_ORDERS = {
    "dataset": [
        "ArXiv ML Papers",
        "BBC News",
        "20 Newsgroups Raw",
        "StackExchange",
        "Wiki Medical",
        "20 Newsgroups Preprocessed",
    ],
    "model": [
        "S³_axial",
        "S³_angular",
        "S³_combined",
        "FASTopic",
        "Top2Vec",
        "BERTopic",
        "CombinedTM",
        "ZeroShotTM",
        "NMF",
        "LDA",
        "ECRTM",
    ],
    "encoder": [
        "GloVe",
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "intfloat/e5-large-v2",
    ],
}

models2colors = {
    "S³_axial": "#5D5DE6",
    "S³_angular": "#1F6AE8",
    "S³_combined": "#4A90E2",
    "Top2Vec": "#77BA99",
    "FASTopic": "#BD4F6C",
    "BERTopic": "#E67B5F",
    "CombinedTM": "#F0CF65",
    "ZeroShotTM": "#DDEDAA",
    "NMF": "#B5B2C2",
    "LDA": "#806D40",
    "ECRTM": "black",
}

encoder2colors = {
    "GloVe": "black",
    "all-MiniLM-L6-v2": "red",
    "all-mpnet-base-v2": "blue",
    "intfloat/e5-large-v2": "green",
}


def set_plt_params(SCALE):
    plt.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "axes.labelsize": 9 * SCALE,
            "xtick.labelsize": 9 * SCALE,
            "ytick.labelsize": 9 * SCALE,
            "legend.fontsize": 9 * SCALE,
            "axes.titlesize": 10 * SCALE,
            "figure.titlesize": 10.5 * SCALE,
            "figure.labelsize": 10.5 * SCALE,
            "axes.linewidth": 1,
        }
    )


def plot_disaggregated(data: pd.DataFrame, metric: str):
    set_plt_params(SCALE=2)
    fig = sns.relplot(
        data=data,
        x="Number of Topics",
        y=metric,
        hue="Model",
        row="Dataset",
        col="Encoder",
        kind="line",
        size="is_target",
        palette=models2colors,
        hue_order=CATEGORY_ORDERS["model"],
        aspect=1,
        height=2.8,
        col_order=CATEGORY_ORDERS["encoder"],
        row_order=CATEGORY_ORDERS["dataset"][:-1],
    )
    fig.set_titles("{col_name}")

    for i in range(len(CATEGORY_ORDERS["encoder"])):
        if i == 1:
            fig.axes[-1, i].set_xlabel("Number of Topics", x=1.1)
        else:
            fig.axes[-1, i].set_xlabel("")

    for i in range(len(CATEGORY_ORDERS["dataset"][:-1])):
        fig.axes[i, 0].set_ylabel(CATEGORY_ORDERS["dataset"][:-1][i])
        if metric == "runtime_s":
            for s in range(fig.axes.shape[1]):
                fig.axes[i, s].set_yscale("log")
    fig.legend.remove()

    fig.fig.legend(
        handles=fig.legend.legendHandles[:12],
        loc=7,
        bbox_to_anchor=(1.01, 0.52),
        frameon=False,
    )
    return fig


ds = load_dataset("kardosdrur/topic-benchmark-results", split="train")
data = ds.to_pandas()
data["model"] = data["model"].replace("S³", "S³_axial")
data["encoder"] = data["encoder"].replace(
    "average_word_embeddings_glove.6B.300d", "GloVe"
)
data = data[data["model"].isin(CATEGORY_ORDERS["model"])]
data = data[data["encoder"].isin(CATEGORY_ORDERS["encoder"])]
data = data[data["seed"] == 42]
data["coherence"] = np.sqrt(data["wec_ex"] * data["wec_in"])
data["is_target"] = np.where(data["model"].str.contains("S³"), 4, 3)
data["encoder"] = data["encoder"].replace(
    "average_word_embeddings_glove.6B.300d", "GloVe"
)
data.rename(
    {
        "dataset": "Dataset",
        "model": "Model",
        "encoder": "Encoder",
        "n_topics": "Number of Topics",
    },
    axis=1,
    inplace=True,
)

data = data[data["Dataset"] != "20 Newsgroups Preprocessed"]
data = data[~data["Model"].isin(["KeyNMF", "GMM"])]

Path("figures").mkdir(exist_ok=True)

for metric in [
    "coherence",
    "wec_in",
    "wec_ex",
    "diversity",
    "runtime_s",
]:
    fig = plot_disaggregated(data, metric)
    metric_name = metric.split("(")[0].lower().replace(" ", "_")
    fig.savefig(
        f"figures/disaggregated_{metric_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
