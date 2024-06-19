from typing import Union

import numpy as np
import pandas as pd

from topic_benchmark.benchmark import BenchmarkEntry, BenchmarkError

METRICS_TO_DISPLAY_NAME = {
    "NPMI Coherence": "C\\textsubscript{NPMI}",
    "Word Embedding Coherence": "WEC\\textsubscript{ex}",
    "Diversity": "Diversity",
    "IWEC": "WEC\\textsubscript{in}",
}

METRICS = [
    "C\\textsubscript{NPMI}",
    "WEC\\textsubscript{ex}",
    "Diversity",
    "WEC\\textsubscript{in}",
]


def produce_header(datasets: list[str]) -> list[str]:
    n_datasets = len(datasets)
    column_align = "l" + ("c" * (n_datasets * len(METRICS)))
    metric_names = " & " + " & ".join(METRICS * n_datasets)
    lines = [
        "\\begin{table*}[p]",
        "\\bgroup",
        "\\resizebox{\\textwidth}{!}{",
        "\\def\\arraystretch{1.3}",
        f"\\begin{{tabular}}{{{column_align}}}",
        "\\toprule",
        " & "
        + " & ".join(
            f"\\multicolumn{{{len(METRICS)}}}{{c}}{{\\textbf{{{dataset}}}}}"
            for dataset in datasets
        )
        + "\\\\",
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
        "\\textit{Best in bold, second best underlined, uncertainty represents $2 \\cdot SD$}",
        "}",
        "\\end{table*}",
    ]


def mean_with_uncertainty(s: pd.Series) -> tuple[float, float]:
    """Returns mean with 2*standard deviation."""
    return s.mean(), 2 * np.std(s)


def format_value(val: tuple[float, float]) -> str:
    m, error = val
    if m < 1:
        error_str = f"{error:.2f}".lstrip("0")
        m_str = f"{m:.2f}"
    else:
        error_str = f"{error:.0f}"
        m_str = f"{m:.0f}"
    return f"{m_str} ± \\textit{{{error_str}}}"


def format_cells(table: pd.DataFrame) -> pd.DataFrame:
    table = table[METRICS]
    bold = dict()
    underline = dict()
    for column in table.columns:
        if column != "runtime_s":
            best_val = table[column].map(lambda val: val[0]).max()
        else:
            best_val = table[column].map(lambda val: val[0]).min()
        bold[column] = best_val
        if column != "runtime_s":
            second_best = (
                table[column].map(lambda val: val[0]).nlargest(n=2).iloc[-1]
            )
        else:
            second_best = (
                table[column].map(lambda val: val[0]).nsmallest(n=2).iloc[-1]
            )
        bold[column] = best_val
        underline[column] = second_best
    records = []
    for model, row in table.iterrows():
        record = {}
        for column, value in row.items():
            formatted_value = format_value(value)
            record[column] = formatted_value
            if value[0] == bold[column]:
                record[column] = f"\\textbf{{{formatted_value}}}"
            if value[0] == underline[column]:
                record[column] = f"\\underline{{{formatted_value}}}"
        records.append(record)
    return pd.DataFrame.from_records(records, index=table.index)[METRICS]


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
    "BBC News",
    "ArXiv ML Papers",
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
    data = data.rename(columns=METRICS_TO_DISPLAY_NAME)
    data = data.groupby(["dataset", "model"])[METRICS].agg(
        mean_with_uncertainty
    )
    groups = dict(list(data.groupby("dataset")))
    # Loading tables in proper order
    group_tables = [groups[dataset] for dataset in DATASET_ORDER]
    n_datasets = len(DATASET_ORDER)
    n_metrics = len(METRICS)
    lines = [
        "\\midrule",
        f"\\textbf{{{encoder_name}}} "
        + " ".join(["&"] * (n_datasets * n_metrics))
        + "\\\\",
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
