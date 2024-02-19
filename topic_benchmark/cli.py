import json
from pathlib import Path
from typing import Optional

from radicli import Arg, Radicli
from sentence_transformers import SentenceTransformer

from topic_benchmark.benchmark import run_benchmark
from topic_benchmark.defaults import default_vectorizer

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
    print("Running Benchmark.")
    entries = run_benchmark(encoder, vectorizer)
    with open(out_path, "w") as out_file:
        for entry in entries:
            out_file.write(json.dumps(entry) + "\n")
    print("DONE")
