import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

CATEGORY_ORDERS = {
    "dataset": [
        "20 Newsgroups Preprocessed",
        "20 Newsgroups Raw",
    ],
    "model": [
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
    "encoder": [
        "average_word_embeddings_glove.6B.300d",
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "intfloat/e5-large-v2",
    ],
}


def count_stop_words(topics: list[list[str]]) -> int:
    stops = set(ENGLISH_STOP_WORDS)
    res = 0
    for topic in topics:
        for word in topic:
            if word in stops:
                res += 1
    return res


def count_nonalphabetical(topics: list[list[str]]) -> int:
    res = 0
    for topic in topics:
        for word in topic:
            if not word.isalpha():
                res += 1
    return res


def plot_stop_words(data: pd.DataFrame) -> go.Figure:
    data = data[data["dataset"] == "20 Newsgroups Raw"]
    data = data.assign(
        n_stop_words=data["topic_descriptions"].map(count_stop_words),
    )
    fig = px.line(
        data,
        color="model",
        x="n_topics",
        y="n_stop_words",
        template="plotly_white",
        category_orders=CATEGORY_ORDERS,
        facet_col="encoder",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig = fig.update_traces(line=dict(width=3))
    fig = fig.update_layout(width=1000, height=800)
    return fig


def plot_nonalphabetical(data: pd.DataFrame) -> go.Figure:
    data = data[data["dataset"] == "20 Newsgroups Raw"]
    data = data.assign(
        n_nonalphabetical=data["topic_descriptions"].map(
            count_nonalphabetical
        ),
    )
    fig = px.line(
        data,
        color="model",
        x="n_topics",
        y="n_nonalphabetical",
        category_orders=CATEGORY_ORDERS,
        facet_col="encoder",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig = fig.update_traces(line=dict(width=3))
    fig = fig.update_layout(width=1000, height=800)
    return fig


def plot_speed(data: pd.DataFrame) -> go.Figure:
    fig = px.box(
        data,
        color="model",
        facet_row="dataset",
        facet_col="encoder",
        y="runtime_s",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        category_orders=CATEGORY_ORDERS,
    )
    fig = fig.update_traces(line=dict(width=3))
    fig = fig.update_layout(width=1000, height=800)
    return fig
