import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Union

import pandas as pd

from topic_benchmark.benchmark.base import BenchmarkEntry


class BenchmarkResults(Mapping):
    """Convenience class wrapping around benchmark results"""

    def __init__(self, encoder_results: dict[str, list[BenchmarkEntry]]):
        self._results = {
            encoder: res for encoder, res in encoder_results.items()
        }

    def __getitem__(self, key: str):
        return self._results[key]

    def __iter__(self):
        return iter(self._results)

    def __len__(self):
        return len(self._results)

    @classmethod
    def load(
        cls,
        results_dir: Union[str, Path] = "results/",
        skip_error: bool = True,
    ):
        results_dir = Path(results_dir)
        files = results_dir.glob("*.jsonl")
        results = {}
        for result_file in files:
            encoder_name = Path(result_file).stem.replace("__", "/")
            entries = []
            with open(result_file) as in_file:
                for line in in_file:
                    if not line.strip():
                        continue
                    # Allows for comments if we want to exclude models.
                    if line.startswith("#"):
                        continue
                    entry = BenchmarkEntry.from_json(line)
                    if (entry.error_message is not None) and skip_error:
                        warnings.warn(
                            f"Not including entry {entry} due to error: {entry.error_message}"
                        )
                        continue
                    entries.append(entry)
            results[encoder_name] = entries
        return cls(results)

    def to_dataframe(self) -> pd.DataFrame:
        records = []
        for encoder in self:
            entries = self[encoder]
            for entry in entries:
                rec = entry.to_dict()
                res = rec.pop("results")
                rec = {"encoder": encoder, **rec, **res}
                records.append(rec)
        df = pd.DataFrame.from_records(records)
        # removing empty columns
        df = df.dropna(axis="columns", how="all")
        return df
