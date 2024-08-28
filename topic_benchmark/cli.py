import json
import warnings
from pathlib import Path
from typing import Optional, Union

from radicli import Arg, Radicli, get_list_converter
from sentence_transformers import SentenceTransformer

from topic_benchmark.benchmark import (BenchmarkEntry, BenchmarkError,
                                       run_benchmark)
from topic_benchmark.defaults import default_vectorizer
from topic_benchmark.registries import encoder_registry
from topic_benchmark.table import produce_full_table


def load_cache(file: Path) -> list[Union[BenchmarkEntry, BenchmarkError]]:
    if not isinstance(file, Path):
        file = Path(file)
    try:
        cached_entries: list[BenchmarkEntry] = []
        with file.open("r") as cache_file:
            print("Loading already completed results")
            for line in cache_file:
                if line.startswith("#"):
                    continue
                line = line.strip()
                if not line:
                    continue
                cached_entries.append(json.loads(line))
        return cached_entries
    except FileNotFoundError:
        with file.open("w") as out_file:
            out_file.write("")
        return []


cli = Radicli()


@cli.command(
    "run",
    out_dir=Arg("--out_dir", "-o", help="Output directory for the results."),
    encoders=Arg(
        "--encoders",
        "-e",
        help="Which encoders should be used for conducting runs?",
    ),
    models=Arg(
        "--models",
        "-m",
        help="What subsection of models should the benchmark be run on.",
        converter=get_list_converter(str, delimiter=","),
    ),
    datasets=Arg(
        "--datasets",
        "-d",
        help="What datasets should the models be evaluated on.",
        converter=get_list_converter(str, delimiter=","),
    ),
    metrics=Arg(
        "--metrics",
        "-t",
        help="What metrics should the models be evaluated on.",
        converter=get_list_converter(str, delimiter=","),
    ),
    seeds=Arg(
        "--seeds",
        "-s",
        help="What seeds should the models be evaluated on.",
        converter=get_list_converter(int, delimiter=","),
    ),
)
def run_cli(
    out_dir: str = "results/",
    encoders: Optional[list[str]] = None,
    models: Optional[list[str]] = None,
    datasets: Optional[list[str]] = None,
    metrics: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
):
    vectorizer = default_vectorizer()

    if encoders is None:
        encoders = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "average_word_embeddings_glove.6B.300d",
            "intfloat/e5-large-v2",
        ]
    print("Loading Encoders.")
    if seeds is None:
        seeds = (42, 43, 44, 45, 46)
    else:
        seeds = tuple(seeds)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    for encoder_name in encoders:
        if encoder_name in encoder_registry:
            encoder = encoder_registry.get(encoder_name)()
        else:
            encoder = SentenceTransformer(encoder_name)
            print(
                f"`{encoder_name}`: encoder model not found in registry. "
                "Loading using `SentenceTransformer`"
            )
        encoder_path_name = encoder_name.replace("/", "__")
        out_path = out_dir.joinpath(f"{encoder_path_name}.jsonl")
        out_path = f"results/{encoder_path_name}.jsonl"
        cached_entries = load_cache(out_path)
        print("--------------------------------------")
        print(f"Running benchmark with {encoder_name}")
        print("--------------------------------------")
        entries = run_benchmark(
            encoder,
            vectorizer,
            models,
            datasets,
            metrics,
            seeds,
            prev_entries=cached_entries,
        )
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
