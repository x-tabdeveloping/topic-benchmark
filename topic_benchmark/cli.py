import json
from pathlib import Path
from typing import Optional

from radicli import Arg, Radicli
from sentence_transformers import SentenceTransformer

from topic_benchmark.benchmark import BenchmarkEntry, run_benchmark
from topic_benchmark.defaults import default_vectorizer

cli = Radicli()


@cli.command(
    "run",
    encoder_model=Arg("--encoder_model", "-e"),
    out_path=Arg("--out_file", "-o"),
)
def run_cli(encoder_model: str = "all-MiniLM-L6-v2", out_path: Optional[str] = None):
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
            done = {(entry["dataset"], entry["model"]) for entry in cached_entries}
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
