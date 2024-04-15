import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS


CATEGORY_ORDERS = {
    "Dataset": [
        "20 Newsgroups Preprocessed",
        "20 Newsgroups Raw",
    ],
    "Model": [
        "S³",
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
    "Top2Vec": "#BD4F6C",
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

    set_plt_params(SCALE=3)

    data = data[data["Dataset"] == "20 Newsgroups Raw"]

    df0 = data[data["Encoder"] == CATEGORY_ORDERS["Encoder"][0]]
    df1 = data[data["Encoder"] == CATEGORY_ORDERS["Encoder"][1]]
    df2 = data[data["Encoder"] == CATEGORY_ORDERS["Encoder"][2]]
    df3 = data[data["Encoder"] == CATEGORY_ORDERS["Encoder"][3]]

    fig, axs = plt.subplots(ncols=4, figsize=(23, 7))

    # fill in the facets
    def fill_facet(df, ax_i):
        axs[ax_i].grid(visible=True, which="major", axis="y", linewidth=0.5, linestyle=':')

        for i, group in df.groupby("Model"):
            group_c = models2colors[group["Model"].tolist()[0]]
            axs[ax_i].plot(
                "Number of Topics",
                "Relative Frequency of Stop Words",
                "-",
                linewidth=3,
                data=group,
                c=group_c,
            )

        axs[ax_i].scatter(
            df["Number of Topics"],
            df["Relative Frequency of Stop Words"],
            c=df["Model"].map(models2colors).tolist(),
            s=40,
        )
        axs[ax_i].set_title(CATEGORY_ORDERS["Encoder"][ax_i])
        axs[ax_i].set_ylim(-0.05, 0.85)
        axs[ax_i].set_yticks(np.arange(0, 1, step=0.2))
        axs[ax_i].set_xticks(np.arange(10, 60, step=10))

        if ax_i > 0:
            axs[ax_i].yaxis.set_ticklabels([])
            for tick in axs[ax_i].yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

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

    set_plt_params(SCALE=3)

    data = data[data["Dataset"] == "20 Newsgroups Raw"]

    df0 = data[data["Encoder"] == CATEGORY_ORDERS["Encoder"][0]]
    df1 = data[data["Encoder"] == CATEGORY_ORDERS["Encoder"][1]]
    df2 = data[data["Encoder"] == CATEGORY_ORDERS["Encoder"][2]]
    df3 = data[data["Encoder"] == CATEGORY_ORDERS["Encoder"][3]]

    fig, axs = plt.subplots(ncols=4, figsize=(23, 7))

    # fill in the facets
    def fill_facet(df, ax_i):
        axs[ax_i].grid(visible=True, which="major", axis="y", linewidth=0.5, linestyle=':')

        for i, group in df.groupby("Model"):
            group_c = models2colors[group["Model"].tolist()[0]]
            axs[ax_i].plot(
                "Number of Topics",
                "Relative Frequency of Nonalphabetical Terms",
                "-",
                linewidth=3,
                data=group,
                c=group_c,
            )

        axs[ax_i].scatter(
            df["Number of Topics"],
            df["Relative Frequency of Nonalphabetical Terms"],
            c=df["Model"].map(models2colors).tolist(),
            s=40,
        )
        axs[ax_i].set_title(CATEGORY_ORDERS["Encoder"][ax_i])
        axs[ax_i].set_ylim(-0.02, 0.32)
        axs[ax_i].set_yticks(np.arange(0, 0.4, step=0.1))
        axs[ax_i].set_xticks(np.arange(10, 60, step=10))

        if ax_i > 0:
            axs[ax_i].yaxis.set_ticklabels([])
            for tick in axs[ax_i].yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

    fill_facet(df0, 0)
    fill_facet(df1, 1)
    fill_facet(df2, 2)
    fill_facet(df3, 3)

    fig.supxlabel("Number of Topics", x=0.45)
    fig.supylabel("RF of Non-Alphabetical Terms", x=-0.004)

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

    set_plt_params(SCALE=3)

    # drop some models
    forbidden_models = ["KeyNMF", "GMM"]
    data = data.query("Model != @forbidden_models")
    MODEL_ORDER = ['NMF', 'LDA', 'S³', 'BERTopic', 'Top2Vec', 'CombinedTM', 'ZeroShotTM']

    data_raw = data[data["Dataset"] == "20 Newsgroups Raw"]
    data_pro = data[data["Dataset"] == "20 Newsgroups Preprocessed"]
    data_raw = data_raw.replace({'20 Newsgroups Raw': 'Raw'})
    data_pro = data_pro.replace({'20 Newsgroups Preprocessed': 'Preprocessed'})

    # facet: model
    # x: n topics, y: processing speed, color: encoder
    def fill_facet(df):

        sns.set_style('whitegrid', {"grid.linestyle": ":",})
        fig = sns.catplot(data=df, x='Model', y='Runtime in Seconds', col='Encoder',
                          kind='box', height=5., aspect=1.1, row='Dataset',
                          whis=(0, 100),
                          width=0.5, linewidth=1.4,
                          order=MODEL_ORDER,
                          margin_titles=True,
                          facet_kws={"despine": False},
                          col_order= ['GloVe',
                                     'all-MiniLM-L6-v2',
                                     'all-mpnet-base-v2', 
                                     'intfloat/e5-large-v2'],
                                     sharey = 'row',
                          palette=models2colors)
        fig.set_titles(col_template='{col_name}', row_template='{row_name}')
        for i in range(4):
            fig.axes[1,i].set_xlabel('')
            fig.axes[0,i].set_ylabel('')
            fig.axes[1,i].set_ylabel('')
            fig.axes[1,i].set_title('')
            fig.axes[1,i].set_xticklabels(fig.axes[1,i].get_xticklabels(), rotation=60, ha='right')
        return fig

    fig = fill_facet(pd.concat([data_raw, data_pro]))
    for ax in fig.axes.flat:
        ax.grid(True, axis='both')
        for _,s in ax.spines.items():
            s.set_linewidth(2)
        try:
            ax.get_xticklabels()[2].set_weight("bold")
        except:
            pass
    fig.figure.supylabel("Runtime (s)", x=-0.004)

    plt.tight_layout()
    return fig


def plot_speed_v2(data):

    set_plt_params(SCALE=3.2)

    # drop some models
    forbidden_models = ["KeyNMF", "GMM"]
    data = data.query("Model != @forbidden_models")
    MODEL_ORDER = CATEGORY_ORDERS["Model"]
    MODEL_ORDER = [m for m in MODEL_ORDER if m not in forbidden_models]

    data_raw = data[data["Dataset"] == "20 Newsgroups Raw"]
    data_pro = data[data["Dataset"] == "20 Newsgroups Preprocessed"]

    fig, axs = plt.subplots(nrows=2, ncols=7, figsize=(28, 10))

    # facet: encoder
    # x: n topics, y: processing speed, color: model
    def fill_facet_upper_row(df, ax_i, row, line_style="-"):

        axs[row][ax_i].grid(visible=True, which="major", axis="y", linewidth=0.5)

        for i, group in df.groupby("Encoder"):
            group_c = encoder2colors[group["Encoder"].tolist()[0]]
            axs[row][ax_i].plot(
                "Number of Topics",
                "Runtime in Seconds",
                line_style,
                linewidth=4,
                data=group,
                c=group_c,
            )
        axs[row][ax_i].set_title(MODEL_ORDER[ax_i])
        axs[row][ax_i].set_ylim(-900, 6000)
        axs[row][ax_i].set_xticks(np.arange(10, 60, step=10))
        axs[row][ax_i].set_yticks(np.arange(0, 7000, step=1000))
        axs[row][ax_i].xaxis.set_tick_params(labelsize=28)
        axs[row][ax_i].yaxis.set_tick_params(labelsize=20)

        if ax_i > 0:
            axs[row][ax_i].yaxis.set_ticklabels([])
            for tick in axs[row][ax_i].yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

    def fill_facet_lower_row(df, ax_i, row, line_style="-"):

        axs[row][ax_i].grid(visible=True, which="major", axis="y", linewidth=0.5)

        for i, group in df.groupby("Encoder"):
            group_c = encoder2colors[group["Encoder"].tolist()[0]]
            axs[row][ax_i].plot(
                "Number of Topics",
                "Runtime in Seconds",
                line_style,
                linewidth=4,
                data=group,
                c=group_c,
            )
        axs[row][ax_i].set_ylim(-900, 15_000)
        axs[row][ax_i].set_xticks(np.arange(10, 60, step=10))
        axs[row][ax_i].set_yticks(np.arange(0, 17_500, step=2500))
        axs[row][ax_i].xaxis.set_tick_params(labelsize=28)
        axs[row][ax_i].yaxis.set_tick_params(labelsize=20)

        if ax_i > 0:
            axs[row][ax_i].yaxis.set_ticklabels([])
            for tick in axs[row][ax_i].yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

    # upper row
    for ax_i, model_tag in enumerate(MODEL_ORDER):
        sub_pro = data_pro[data_pro["Model"] == model_tag]
        fill_facet_upper_row(sub_pro, ax_i, row=0)

    # lower row
    for ax_i, model_tag in enumerate(MODEL_ORDER):
        sub_raw = data_raw[data_raw["Model"] == model_tag]
        fill_facet_lower_row(sub_raw, ax_i, row=1)

    axs[0,0].set_ylabel('Preprocessed')
    axs[1,0].set_ylabel('Raw')
    fig.supxlabel("Number of Topics", x=0.52)
    fig.supylabel("Runtime (s)", x=0.05)

    
    legend_handles = [
        Patch(facecolor=encoder2colors[encoder], label=encoder)
        for encoder in CATEGORY_ORDERS["Encoder"]
    ]
    plt.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        bbox_to_anchor=(0.32, 2.6),
        borderaxespad=0.,
        ncol=4
    )

    plt.tight_layout()

    return fig
