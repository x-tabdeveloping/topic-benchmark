import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

CATEGORY_ORDERS = {
    "Dataset": [
        "ArXiv ML Papers",
        "BBC News",
        "20 Newsgroups Raw",
        "20 Newsgroups Preprocessed",
    ],
    "Model": [
        "S³",
        "FASTopic",
        "Top2Vec",
        "BERTopic",
        "CombinedTM",
        "ZeroShotTM",
        "NMF",
        "LDA",
    ],
    "Encoder": [
        "GloVe",
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "intfloat/e5-large-v2",
    ],
}

models2colors = {
    "S³": "#5D5DE6",
    "Top2Vec": "#77BA99",
    "FASTopic": "#BD4F6C",
    "BERTopic": "#D7816A",
    "CombinedTM": "#F0CF65",
    "ZeroShotTM": "#DDEDAA",
    "NMF": "#B5B2C2",
    "LDA": "#806D40",
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
    data["Model"] = data["model"]
    data["Dataset"] = data["dataset"]
    data["Encoder"] = data["encoder"].replace(
        "average_word_embeddings_glove.6B.300d", "GloVe"
    )
    data["Number of Topics"] = data["n_topics"]
    set_plt_params(SCALE=2)
    # sns.set_style('whitegrid', {"grid.linestyle": ":",})
    data = data[data["Dataset"] != "20 Newsgroups Preprocessed"]
    data = data[~data["Model"].isin(["KeyNMF", "GMM"])]

    data["is_target"] = np.where(data["Model"] == "S³", 4, 2)

    fig = sns.relplot(
        data=data,
        x="Number of Topics",
        y=metric,
        row="Dataset",
        hue="Model",
        col="Encoder",
        size="is_target",
        kind="line",
        palette=models2colors,
        hue_order=CATEGORY_ORDERS["Model"],
        aspect=1,
        height=3,
        col_order=CATEGORY_ORDERS["Encoder"],
        row_order=CATEGORY_ORDERS["Dataset"][:-1],
    )
    fig.set_titles("{col_name}")
    for i in range(len(CATEGORY_ORDERS["Dataset"][:-1])):
        fig.axes[i, 0].set_ylabel(CATEGORY_ORDERS["Dataset"][:-1][i])
    fig.legend.remove()

    fig.fig.legend(
        handles=fig.legend.legendHandles[:8],
        loc=7,
        bbox_to_anchor=(1.01, 0.52),
        frameon=False,
    )
    return fig


STOPS = set(ENGLISH_STOP_WORDS)


def stop_word_rel_freq(topic_descriptions: list[list[str]]) -> float:
    words = list(itertools.chain.from_iterable(topic_descriptions))
    total = len(words)
    n_stop = 0
    for word in words:
        if word in STOPS:
            n_stop += 1
    return n_stop / total


def rel_freq_nonalphabetical(topics: list[list[str]]) -> float:
    res = 0
    total = 0
    for topic in topics:
        for word in topic:
            total += 1
            if not word.isalpha():
                res += 1
    return res / total


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

out_dir = Path("figures")
out_dir.mkdir(exist_ok=True)
data = pd.DataFrame.from_records(entries)
data = data[data["dataset"] != "20 Newsgroups Preprocessed"]
data["stop_freq"] = data["topic_descriptions"].map(stop_word_rel_freq)
data["nonalpha_freq"] = data["topic_descriptions"].map(
    rel_freq_nonalphabetical
)
for metric in [
    "IWEC",
    "Word Embedding Coherence",
    "Diversity",
    "runtime_s",
    "stop_freq",
    "nonalpha_freq",
]:
    fig = plot_disaggregated(data, metric)
    metric_name = metric.lower().replace(" ", "_")
    out_path = out_dir.joinpath(f"disaggregated_{metric_name}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
