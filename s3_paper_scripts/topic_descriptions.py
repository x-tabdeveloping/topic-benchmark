import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_ORDER = [
    "S³",
    "FASTopic",
    "Top2Vec",
    "BERTopic",
    "CombinedTM",
    "ZeroShotTM",
    "NMF",
    "LDA",
]

EMBEDDING_ORDER = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "average_word_embeddings_glove.6B.300d",
    "intfloat/e5-large-v2",
]

results_folder = Path("results/")
files = results_folder.glob("*.jsonl")
out_path = Path("tables/", "topic_description.tex")
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
    (data["dataset"] == "20 Newsgroups Raw")
    & (data["n_topics"] == 20)
    & data["model"].isin(MODEL_ORDER)
    & data["encoder"].isin(EMBEDDING_ORDER)
)
data = data[include]


lines = []
lines.extend(
    [
        "\\section{Topic Descriptions from Qualitative Analyses on 20Newsgroups}",
        "\\label{sec:topic_desc}",
    ]
)
for encoder in EMBEDDING_ORDER:
    lines.append(
        f"\\subsection{{{encoder}}}\\vspace{{2mm}}",
    )
    for model in MODEL_ORDER:
        lines.extend(
            [
                "\\hrule\\vspace{2mm}",
                "\\noindent",
                f"\\textbf{{{model}}}\\vspace{{2mm}}\\\\",
                "\\vspace{2mm}",
                "\\noindent",
            ]
        )
        model = model.replace("_", "\\_")
        model_data = data[
            (data["model"] == model) & (data["encoder"] == encoder)
        ]
        topic_desc = model_data["topic_descriptions"].iloc[0]
        for i_topic, topic in enumerate(topic_desc):
            line = f"\\textbf{{{i_topic}}} - " + ", ".join(topic) + "\\\\"
            line = line.replace("_", "\\_")
            lines.append(line)

with out_path.open("w") as out_file:
    out_file.write("\n".join(lines))
