import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS


CATEGORY_ORDERS = {
    "Dataset": [
        "20 Newsgroups Preprocessed",
        "20 Newsgroups Raw",
    ],
    "Model": [
        "NMF",
        "LDA",
        "S³",
        "KeyNMF",
        "GMM",
        "Top2Vec",
        "BERTopic",
        "CombinedTM",
        "ZeroShotTM",
    ],
    "Encoder": [
        "GloVe",
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "intfloat/e5-large-v2",
    ],
}

SCALE = 2.5

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "Times New Roman",
        "font.serif": "serif",
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

models2colors = {
    "NMF": "#66C5CC",
    "LDA": "#F6CF71",
    "S³": "#F89C74",
    "KeyNMF": "#DCB0F2",
    "GMM": "#87C55F",
    "Top2Vec": "#9EB9F3",
    "BERTopic": "#FE88B1",
    "CombinedTM": "#C9DB74",
    "ZeroShotTM": "#8BE0A4",
}


def rel_freq_stop_words(topics: list[list[str]]) -> int:
    stops = set(ENGLISH_STOP_WORDS)
    res = 0
    total = 0
    for topic in topics:
        for word in topic:
            total += 1
            if word in stops:
                res += 1
    return res / total


def rel_freq_nonalphabetical(topics: list[list[str]]) -> int:
    res = 0
    total = 0
    for topic in topics:
        for word in topic:
            total += 1
            if not word.isalpha():
                res += 1
    return res / total


def preprocess_for_plotting(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data = data.rename(
        columns={
            "model": "Model",
            "encoder": "Encoder",
            "dataset": "Dataset",
            "n_topics": "Number of Topics",
            "runtime_s": "Runtime in Seconds",
        }
    )
    data["Relative Frequency of Nonalphabetical Terms"] = data[
        "topic_descriptions"
    ].map(rel_freq_nonalphabetical)
    data["Relative Frequency of Stop Words"] = data["topic_descriptions"].map(
        rel_freq_stop_words
    )
    data["Encoder"] = data["Encoder"].replace(
        {"average_word_embeddings_glove.6B.300d": "GloVe"}
    )
    return data


def plot_stop_words(data):

    data = data[data["Dataset"] == "20 Newsgroups Raw"]

    df0 = data[data["Encoder"] == CATEGORY_ORDERS["Encoder"][0]]
    df1 = data[data["Encoder"] == CATEGORY_ORDERS["Encoder"][1]]
    df2 = data[data["Encoder"] == CATEGORY_ORDERS["Encoder"][2]]
    df3 = data[data["Encoder"] == CATEGORY_ORDERS["Encoder"][3]]

    fig, axs = plt.subplots(ncols=4, figsize=(20, 5))

    for i, group in df0.groupby("Model"):
        group_c = models2colors[group["Model"].tolist()[0]]
        axs[0].plot(
            "Number of Topics",
            "Relative Frequency of Stop Words",
            "-",
            data=group,
            c=group_c,
        )

    # fill in the facets
    def fill_facet(df, ax_i):
        axs[ax_i].grid(visible=True, which="major", axis="y", linewidth=0.3)

        for i, group in df.groupby("Model"):
            group_c = models2colors[group["Model"].tolist()[0]]
            axs[ax_i].plot(
                "Number of Topics",
                "Relative Frequency of Stop Words",
                "-",
                data=group,
                c=group_c,
            )

        axs[ax_i].scatter(
            df["Number of Topics"],
            df["Relative Frequency of Stop Words"],
            c=df["Model"].map(models2colors).tolist(),
        )
        axs[ax_i].set_title(CATEGORY_ORDERS["Encoder"][ax_i])
        axs[ax_i].set_ylim(-0.05, 0.85)
        axs[ax_i].set_yticks(np.arange(0, 1, step=0.2))
        axs[ax_i].set_xticks(np.arange(10, 60, step=10))

    fill_facet(df0, 0)
    fill_facet(df1, 1)
    fill_facet(df2, 2)
    fill_facet(df3, 3)

    fig.supxlabel("Number of Topics", x=0.45)
    fig.supylabel("RF of Stop Words", x=-0.004)

    legend_handles = [
        Patch(facecolor=models2colors[model], label=model)
        for model in CATEGORY_ORDERS["Model"]
    ]
    plt.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        frameon=False,
    )
    plt.tight_layout()

    return fig


def plot_nonalphabetical(data):

    data = data[data["Dataset"] == "20 Newsgroups Raw"]

    df0 = data[data["Encoder"] == CATEGORY_ORDERS["Encoder"][0]]
    df1 = data[data["Encoder"] == CATEGORY_ORDERS["Encoder"][1]]
    df2 = data[data["Encoder"] == CATEGORY_ORDERS["Encoder"][2]]
    df3 = data[data["Encoder"] == CATEGORY_ORDERS["Encoder"][3]]

    fig, axs = plt.subplots(ncols=4, figsize=(20, 5))

    for i, group in df0.groupby("Model"):
        group_c = models2colors[group["Model"].tolist()[0]]
        axs[0].plot(
            "Number of Topics",
            "Relative Frequency of Nonalphabetical Terms",
            "-",
            data=group,
            c=group_c,
        )

    # fill in the facets
    def fill_facet(df, ax_i):
        axs[ax_i].grid(visible=True, which="major", axis="y", linewidth=0.3)

        for i, group in df.groupby("Model"):
            group_c = models2colors[group["Model"].tolist()[0]]
            axs[ax_i].plot(
                "Number of Topics",
                "Relative Frequency of Nonalphabetical Terms",
                "-",
                data=group,
                c=group_c,
            )

        axs[ax_i].scatter(
            df["Number of Topics"],
            df["Relative Frequency of Nonalphabetical Terms"],
            c=df["Model"].map(models2colors).tolist(),
        )
        axs[ax_i].set_title(CATEGORY_ORDERS["Encoder"][ax_i])
        axs[ax_i].set_ylim(-0.02, 0.32)
        axs[ax_i].set_yticks(np.arange(0, 0.4, step=0.1))
        axs[ax_i].set_xticks(np.arange(10, 60, step=10))

    fill_facet(df0, 0)
    fill_facet(df1, 1)
    fill_facet(df2, 2)
    fill_facet(df3, 3)

    fig.supxlabel("Number of Topics", x=0.45)
    fig.supylabel("RF of Non-alphabetical Terms", x=-0.004)

    legend_handles = [
        Patch(facecolor=models2colors[model], label=model)
        for model in CATEGORY_ORDERS["Model"]
    ]
    plt.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        frameon=False,
    )
    plt.tight_layout()

    return fig


def plot_speed(data):
    return plt.figure()
