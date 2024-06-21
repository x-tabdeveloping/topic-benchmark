import json
from pathlib import Path

import numpy as np
import pandas as pd


def hdi_97(a):
    return np.quantile(a, 0.975)


def hdi_2(a):
    return np.quantile(a, 0.025)


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


def percent_s3(data: pd.DataFrame):
    data = data.set_index("model")
    data = data[["runtime_s"]]
    data = data.assign(
        percent_s3=(data["runtime_s"] / data.loc["S³"]["runtime_s"])
    )
    return data


data = pd.DataFrame.from_records(entries)
data = data[data["dataset"] != "20 Newsgroups Preprocessed"]
data = data[["model", "runtime_s", "n_topics", "dataset", "encoder"]]
# data = data[~data["model"].isin(["NMF", "LDA"])]
data = data.groupby(["n_topics", "dataset", "encoder"]).apply(percent_s3)
data["percent_slower"] = data["percent_s3"] - 1
data = data.reset_index()

summary = data.groupby("model")["percent_slower"].agg(["mean", hdi_97, hdi_2])
summary["positive"] = summary["hdi_97"] - summary["mean"]
summary["negative"] = summary["mean"] - summary["hdi_2"]

out_path = Path("tables/speed.tex")
out_path.parent.mkdir(exist_ok=True)
with out_path.open("w") as out_file:
    out_file.write(
        "\\textbf{Model} & \\textbf{Runtime Difference From S\\textsuperscript{3}} \\\\ \n"
    )
    out_file.write("\\midrule \n")
    for model, row in (
        (summary * 100)
        .sort_values("mean")
        .drop(columns=["positive", "negative"])
        .iterrows()
    ):
        lower = np.format_float_positional(
            row["hdi_2"], precision=3, unique=False, fractional=False, trim="k"
        ).removesuffix(".")
        upper = np.format_float_positional(
            row["hdi_97"],
            precision=3,
            unique=False,
            fractional=False,
            trim="k",
        ).removesuffix(".")
        val = np.format_float_positional(
            row["mean"], precision=3, unique=False, fractional=False, trim="k"
        ).removesuffix(".")
        out_file.write(f"{model} & {val}\% [{lower}, {upper}] \\\\ \n")
