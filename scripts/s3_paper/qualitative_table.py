from itertools import islice

import pandas as pd
from datasets import load_dataset


def format_model_name(model: str) -> str:
    if model == "S³":
        model = "S³_axial"
    if model.startswith("S³"):
        _, method = model.split("_")
        method_short = method.lower()[:3]
        model = f"$S^3_{{{method_short}}}$"
    return model


ds = load_dataset("kardosdrur/topic-benchmark-results", split="train")
models = [
    "$S^3_{axi}$",
    "$S^3_{ang}$",
    "$S^3_{com}$",
    "Top2Vec",
    "FASTopic",
    "ECRTM",
    "BERTopic",
    "CombinedTM",
    "ZeroShotTM",
    "LDA",
    "NMF",
]
encoders = [
    "average_word_embeddings_glove.6B.300d",
    "all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
    "intfloat/e5-large-v2",
]
noncontextual = ["ECRTM", "LDA", "NMF"]
data = ds.to_pandas()
data = data[data["encoder"].isin(encoders)]
data["model"] = data["model"].map(format_model_name)
data = data[data["model"].isin(models)]
data["encoder"] = data["encoder"].str.replace("_", "\\_")
data = data[data["seed"] == 42]
data = data[data["dataset"] == "20 Newsgroups Raw"]
data = data[data["n_topics"] == 20]
data["encoder"] = data["encoder"].mask(
    data["model"].isin(noncontextual), "No Encoder"
)

lines = []
lines.extend(
    [
        "\\section{Topic Descriptions from Qualitative Analyses on 20Newsgroups}",
        "\\label{sec:topic_desc}",
    ]
)
for encoder, encoder_data in data.groupby("encoder"):
    lines.append(
        f"\\subsection{{{encoder}}}\\vspace{{2mm}}",
    )
    for model, model_data in encoder_data.groupby("model"):
        lines.extend(
            [
                f"\\subsubsection{{{model}}}",
                "\\noindent",
            ]
        )
        topic_desc = model_data["topic_descriptions"].iloc[0]
        for i_topic, topic in enumerate(topic_desc):
            line = f"\\textbf{{{i_topic}}} - " + ", ".join(topic) + "\\\\"
            line = line.replace("_", "\\_")
            lines.append(line)

print("\n" + "\n".join(lines))
