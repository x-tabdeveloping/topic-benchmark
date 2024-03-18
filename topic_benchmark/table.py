from typing import Union

import pandas as pd

from topic_benchmark.benchmark import BenchmarkEntry, BenchmarkError


def produce_header(datasets: list[str]) -> list[str]:
    n_datasets = len(datasets)
    column_align = "l" + ("c" * (n_datasets * 3))
    midrules = []
    for i in range(n_datasets):
        start = (i + 1) * 3 - 1
        end = start + 2
        midrule = f"\\cmidrule(l{{3pt}}r{{3pt}}){{{start}-{end}}}"
        midrules.append(midrule)
    metric_names = " & " + " & ".join(
        ["NPMI", "WEC", "Diversity"] * n_datasets
    )
    lines = [
        "\\begin{table}[h]",
        "\\bgroup",
        "\\resizebox{\\textwidth}{!}{",
        "\\def\\arraystretch{1.3}",
        f"\\begin{{tabular}}{{{column_align}}}",
        "\\toprule",
        " & "
        + " & ".join(
            f"\\multicolumn{{3}}{{c}}{{\\textbf{{{dataset}}}}}"
            for dataset in datasets
        )
        + "\\\\",
        " ".join(midrules),
        metric_names + "\\\\",
        "\\midrule",
    ]
    return lines


def produce_footer() -> list[str]:
    return [
        "\\bottomrule",
        "\\label{table:evaluation}",
        "\\end{tabular}",
        "}",
        "\\egroup",
        "\\caption{ Coherence and Diversity of Topics over Multiple Datasets \\\\",
        "\\textit{Best in bold, second best underlined}",
        "}",
        "\\end{table}",
    ]


def format_cells(table: pd.DataFrame) -> pd.DataFrame:
    table = table[["NPMI Coherence", "Word Embedding Coherence", "Diversity"]]
    bold = {column: table[column].max() for column in table.columns}
    underline = {
        column: table[column].nlargest(n=2).iloc[-1]
        for column in table.columns
    }
    records = []
    for model, row in table.iterrows():
        record = {}
        for column, value in row.items():
            formatted_value = f"{value:.3f}"
            record[column] = formatted_value
            if value == bold[column]:
                record[column] = f"\\textbf{{{formatted_value}}}"
            if value == underline[column]:
                record[column] = f"\\underline{{{formatted_value}}}"
        records.append(record)
    return pd.DataFrame.from_records(records, index=table.index)[
        ["NPMI Coherence", "Word Embedding Coherence", "Diversity"]
    ]


MODEL_ORDER = [
    "S³",
    "KeyNMF",
    "GMM",
    "Top2Vec",
    "BERTopic",
    "CombinedTM",
    "ZeroShotTM",
    "NMF",
    "LDA",
]

EMBEDDING_ORDER = [
    "average_word_embeddings_glove.6B.300d",
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "intfloat/e5-large-v2",
]

DATASET_ORDER = [
    "20 Newsgroups Preprocessed",
    "20 Newsgroups Raw",
]


def produce_body(groups: list[pd.DataFrame]) -> list[str]:
    groups = [group.reset_index().set_index("model") for group in groups]
    models = set()
    for group in groups:
        models |= set(group.index)
    models = [model for model in MODEL_ORDER if model in models]
    formatted = [format_cells(group) for group in groups]
    lines = []
    for model in models:
        row = []
        for group in formatted:
            try:
                row.extend(group.loc[model])
            except KeyError:
                row.extend(["NA"] * 3)
        lines.append(" & ".join([model, *row]) + "\\\\")
    return lines


def produce_encoder_rows(
    entries: list[Union[BenchmarkEntry, BenchmarkError]],
    encoder_name: str,
) -> list[str]:
    """Produces lines in a table for a single embedding model."""
    encoder_name = encoder_name.replace(
        "_", "\\_"
    )  # Escaping underscores for LaTex
    entries = [entry for entry in entries if "error_message" not in entry]
    data = pd.DataFrame.from_records(entries)
    data = data.join(pd.json_normalize(data["results"]))
    data = data.groupby(["dataset", "model"])[
        [
            "NPMI Coherence",
            "Word Embedding Coherence",
            "Diversity",
        ]
    ].mean()
    groups = dict(list(data.groupby("dataset")))
    # Loading tables in proper order
    group_tables = [groups[dataset] for dataset in DATASET_ORDER]
    lines = [
        "\\midrule",
        f"\\textbf{{{encoder_name}}} & & & & & & \\\\",
        "\\midrule",
        *produce_body(group_tables),
    ]
    return lines


def produce_full_table(
    encoder_entries: dict[str, Union[BenchmarkEntry, BenchmarkError]],
) -> str:
    """Produces full table for all encoder models."""
    lines = [*produce_header(DATASET_ORDER)]
    for encoder in EMBEDDING_ORDER:
        if encoder not in encoder_entries:
            continue
        lines.extend(produce_encoder_rows(encoder_entries[encoder], encoder))
    lines.extend(produce_footer())
    return "\n".join(lines)
