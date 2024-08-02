import json
import warnings
from pathlib import Path
from typing import Optional

from radicli import Arg, Radicli, get_list_converter
from sentence_transformers import SentenceTransformer

from topic_benchmark.benchmark import BenchmarkEntry, run_benchmark
from topic_benchmark.defaults import default_vectorizer
from topic_benchmark.registries import encoder_registry
from topic_benchmark.table import produce_full_table

cli = Radicli()


@cli.command(
    "run",
    encoder_model=Arg("--encoder_model", "-e"),
    out_path=Arg("--out_file", "-o"),
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
    encoder_model: str = "all-MiniLM-L6-v2",
    out_path: Optional[str] = None,
    models: Optional[list[str]] = None,
    datasets: Optional[list[str]] = None,
    metrics: Optional[list[str]] = None,
    seeds: Optional[list[int]] = None,
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

    if seeds is None:
        seeds = (42,)
    else:
        seeds = tuple(seeds)
    out_dir = Path(out_path).parent
    out_dir.mkdir(exist_ok=True, parents=True)
    try:
        with open(out_path, "r") as cache_file:
            print("Loading already completed results")
            cached_entries: list[BenchmarkEntry] = [
                json.loads(line) for line in cache_file
            ]
    except FileNotFoundError:
        cached_entries = []
        with open(out_path, "w") as out_file:
            out_file.write("")
    print("Running Benchmark.")
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
