from pathlib import Path

import pandas as pd
import plotly.express as px
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS


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


def produce_figures(results_file: str, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    data = pd.read_json(results_file, orient="records", lines=True)
    data = data[data["dataset"] == "20 Newsgroups Raw"]
    data["n_stop_words"] = data["topic_descriptions"].map(count_stop_words)
    data["n_nonalphabetical"] = data["topic_descriptions"].map(
        count_nonalphabetical
    )
    print("Producing Stop Words figure.")
    fig = px.line(
        data,
        color="model",
        x="n_topics",
        y="n_stop_words",
        template="plotly_white",
    )
    fig = fig.update_traces(line=dict(width=3))
    fig = fig.update_layout(width=1000, height=800)
    fig.write_image(out_dir.joinpath("stop_words.png"), scale=2)
    print("Producing Nonalphabetical figure.")
    fig = px.line(
        data,
        color="model",
        x="n_topics",
        y="n_nonalphabetical",
        template="plotly_white",
    )
    fig = fig.update_traces(line=dict(width=3))
    fig = fig.update_layout(width=1000, height=800)
    fig.write_image(out_dir.joinpath("n_nonalphabetical.png"), scale=2)
    print("Producing Speed figure.")
    fig = px.line(
        data,
        color="model",
        facet_col="dataset",
        x="n_topics",
        y="runtime_s",
        template="plotly_white",
    )
    fig = fig.update_traces(line=dict(width=3))
    fig = fig.update_layout(width=1000, height=800)
    fig.write_image(out_dir.joinpath("speed.png"), scale=2)
