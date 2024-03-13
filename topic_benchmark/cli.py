import json
from pathlib import Path
from typing import Optional

from radicli import Arg, Radicli
from sentence_transformers import SentenceTransformer

from topic_benchmark.benchmark import BenchmarkEntry, run_benchmark
from topic_benchmark.defaults import default_vectorizer
from topic_benchmark.figures import produce_figures
from topic_benchmark.table import produce_latex_table

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
    encoder = SentenceTransformer(encoder_model)
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
    results_file=Arg(help="JSONL file containing benchmark results."),
    out_path=Arg("--out_file", "-o"),
)
def make_table(results_file: str, out_path: Optional[str] = None):
    with open(results_file) as in_file:
        # Allows for comments if we want to exclude models.
        entries = [
            json.loads(line) for line in in_file if not line.startswith("#")
        ]
    table = produce_latex_table(entries)
    if out_path is None:
        print(table)
    else:
        with open(out_path, "w") as out_file:
            out_file.write(table)


@cli.command(
    "figures",
    results_file=Arg(help="JSONL file containing benchmark results."),
    out_dir=Arg(
        "--out_dir", "-o", help="Directory where the figures should be placed."
    ),
)
def make_figures(results_file: str, out_dir: str = "figures"):
    produce_figures(results_file, out_dir)
