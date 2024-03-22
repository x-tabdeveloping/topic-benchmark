import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


def plot_stop_words(data: pd.DataFrame) -> go.Figure:
    data = data[data["Dataset"] == "20 Newsgroups Raw"]
    fig = px.line(
        data,
        color="Model",
        x="Number of Topics",
        y="Relative Frequency of Stop Words",
        template="plotly_white",
        category_orders=CATEGORY_ORDERS,
        facet_col="Encoder",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig = fig.update_traces(line=dict(width=3))
    fig = fig.update_layout(width=900, height=400)
    return fig


def plot_nonalphabetical(data: pd.DataFrame) -> go.Figure:
    data = data[data["Dataset"] == "20 Newsgroups Raw"]
    fig = px.line(
        data,
        color="Model",
        x="Number of Topics",
        y="Relative Frequency of Nonalphabetical Terms",
        category_orders=CATEGORY_ORDERS,
        facet_col="Encoder",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig = fig.update_traces(line=dict(width=3))
    fig = fig.update_layout(width=1400, height=600)
    return fig


def plot_speed(data: pd.DataFrame) -> go.Figure:
    fig = px.box(
        data,
        color="Model",
        facet_row="Dataset",
        facet_col="Encoder",
        y="Runtime in Seconds",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        category_orders=CATEGORY_ORDERS,
    )
    fig = fig.update_traces(line=dict(width=3))
    fig = fig.update_layout(width=1000, height=600, margin=dict(b=0))
    return fig
