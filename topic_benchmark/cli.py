import json
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
from radicli import Arg, Radicli
from sentence_transformers import SentenceTransformer

from topic_benchmark.benchmark import BenchmarkEntry, run_benchmark
from topic_benchmark.defaults import default_vectorizer
from topic_benchmark.figures_plt import (
    plot_nonalphabetical,
    plot_speed,
    plot_speed_v2,
    plot_stop_words,
    preprocess_for_plotting,
)
from topic_benchmark.registries import encoder_registry
from topic_benchmark.table import produce_full_table

cli = Radicli()


@cli.command(
    "run",
    encoder_model=Arg("--encoder_model", "-e"),
    out_path=Arg("--out_file", "-o"),
)
def run_cli(
    encoder_model: str = "all-MiniLM-L6-v2", out_path: Optional[str] = None
):
    vectorizer = default_vectorizer()

    print("Loading Encoder.")
    if encoder_model in encoder_registry:
        encoder = encoder_registry.get(encoder_model)()
    else:
        encoder = SentenceTransformer(encoder_model)
        print(
            f"`{encoder_model}`: encoder model not found in registry. "
            "Loading using `SentenceTransformer`"
        )

    if out_path is None:
        encoder_path_name = encoder_model.replace("/", "__")
        out_path = f"results/{encoder_path_name}.jsonl"

    out_dir = Path(out_path).parent
    out_dir.mkdir(exist_ok=True, parents=True)
    try:
        with open(out_path, "r") as cache_file:
            print("Loading already completed results")
            cached_entries: list[BenchmarkEntry] = [
                json.loads(line) for line in cache_file
            ]
            done = {
                (entry["dataset"], entry["model"], entry["n_topics"])
                for entry in cached_entries
            }
    except FileNotFoundError:
        with open(out_path, "w") as out_file:
            out_file.write("")
        done = set()
    print("Running Benchmark.")
    entries = run_benchmark(encoder, vectorizer, done=done)
    for entry in entries:
        with open(out_path, "a") as out_file:
            out_file.write(json.dumps(entry) + "\n")
    print("DONE")


@cli.command(
    "table",
    results_folder=Arg(
        help="Folder containing results for all embedding models."
    ),
    out_path=Arg("--out_file", "-o"),
)
def make_table(
    results_folder: str = "results/", out_path: Optional[str] = None
):
    results_folder = Path(results_folder)
    files = results_folder.glob("*.jsonl")
    encoder_entries = dict()
    for result_file in files:
        encoder_name = Path(result_file).stem.replace("__", "/")
        with open(result_file) as in_file:
            # Allows for comments if we want to exclude models.
            entries = [
                json.loads(line)
                for line in in_file
                if not line.startswith("#")
            ]
        encoder_entries[encoder_name] = entries
    table = produce_full_table(encoder_entries)
    if out_path is None:
        print(table)
    else:
        with open(out_path, "w") as out_file:
            out_file.write(table)


@cli.command(
    "figures",
    results_folder=Arg(
        help="Folder containing results for all embedding models."
    ),
    out_dir=Arg(
        "--out_dir", "-o", help="Directory where the figures should be placed."
    ),
)
def make_figures(
    results_folder: str = "results/",
    out_dir: str = "figures",
):
    results_folder = Path(results_folder)
    files = results_folder.glob("*.jsonl")
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    dfs = []
    for file in files:
        file = Path(file)
        df = pd.read_json(file, orient="records", lines=True)
        df["encoder"] = file.stem.replace("__", "/")
        dfs.append(df)
    data = pd.concat(dfs)
    data = preprocess_for_plotting(data)
    figures = {
        "n_nonalphabetical": plot_nonalphabetical,
        "stop_words": plot_stop_words,
        "speed_v2": plot_speed_v2,
        "speed": plot_speed,
    }
    for figure_name, produce in figures.items():
        fig = produce(data)
        out_path = out_dir.joinpath(f"{figure_name}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
