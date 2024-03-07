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
    metric_names = " & " + " & ".join(["NPMI", "WEC", "Diversity"] * n_datasets)
    lines = [
        "\\begin{table}[h]",
        "\\bgroup",
        "\\resizebox{\\textwidth}{!}{",
        "\\def\\arraystretch{1.3}",
        f"\\begin{{tabular}}{{{column_align}}}",
        "\\toprule",
        " & "
        + " & ".join(
            f"\\multicolumn{{3}}{{c}}{{\\textbf{{{dataset}}}}}" for dataset in datasets
        )
        + "\\\\",
        " ".join(midrules),
        metric_names + "\\\\",
        "\\midrule",
    ]
    return lines


def group_by_dataset(entries: list[BenchmarkEntry]) -> dict[str, pd.DataFrame]:
    res = {}
    for entry in entries:
        if entry["dataset"] not in res:
            res[entry["dataset"]] = []
        res[entry["dataset"]].append({"model": entry["model"], **entry["results"]})
    return {
        dataset: pd.DataFrame.from_records(records).set_index("model")
        for dataset, records in res.items()
    }


def format_cells(table: pd.DataFrame) -> pd.DataFrame:
    bold = {column: table[column].max() for column in table.columns}
    underline = {
        column: table[column].nlargest(n=2).iloc[-1] for column in table.columns
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


def produce_body(groups: list[pd.DataFrame]) -> list[str]:
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


def produce_latex_table(entries: list[Union[BenchmarkEntry, BenchmarkError]]) -> str:
    entries = [entry for entry in entries if "error_message" not in entry]
    groups = group_by_dataset(entries)
    group_names = list(groups.keys())
    group_tables = [groups[group] for group in group_names]
    lines = [
        *produce_header(group_names),
        *produce_body(group_tables),
        *produce_footer(),
    ]
    return "\n".join(lines)
