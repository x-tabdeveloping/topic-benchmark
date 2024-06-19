import itertools
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
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

data = pd.DataFrame.from_records(entries)
data = data[data["dataset"] != "20 Newsgroups Preprocessed"]
data = data[["model", "topic_descriptions", "n_topics"]]

data["stop_freq"] = data["topic_descriptions"].map(stop_word_rel_freq)

Path("figures").mkdir(exist_ok=True)
models = [
    "BERTopic",
    "NMF",
    "LDA",
    "CombinedTM",
    "ZeroShotTM",
    "Top2Vec",
    "S³",
]
fig = go.Figure()
fig = fig.add_box(
    x=data["model"],
    y=data["stop_freq"],
    fillcolor="white",
    marker=dict(color="black"),
)
fig = fig.update_layout(width=400, height=300, template="plotly_white")
fig = fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
fig = fig.update_xaxes(categoryorder="array", categoryarray=models)
fig = fig.update_layout(
    font=dict(family="Times New Roman", size=14, color="black")
)
fig.write_image("figures/stop_freq.png", scale=2)
fig.show()
