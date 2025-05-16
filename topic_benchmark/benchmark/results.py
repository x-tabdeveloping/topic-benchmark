import warnings
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from tqdm import tqdm

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

    def select(
        self,
        encoders: Optional[list[str]] = None,
        models: Optional[list[str]] = None,
        datasets: Optional[list[str]] = None,
        n_topics: Optional[list[str]] = None,
    ) -> "BenchmarkResults":
        new_res = defaultdict(list)
        for encoder in self.keys():
            if (encoders is not None) and (encoder not in encoders):
                continue
            for entry in self[encoder]:
                if (models is not None) and (entry.model not in models):
                    continue
                if (datasets is not None) and (entry.dataset not in datasets):
                    continue
                if (n_topics is not None) and (entry.n_topics not in n_topics):
                    continue
                new_res[encoder].append(entry)
        return type(self)(new_res)

    def plot_topics_with_images(
        self,
        n_cols: int = 6,
        grid_size: int = 4,
        image_size: int = 1200,
        scale_factor: float = 0.25,
    ):
        try:
            import plotly.graph_objects as go
        except (ImportError, ModuleNotFoundError) as e:
            raise ModuleNotFoundError(
                "Please install plotly if you intend to use plots in Turftopic."
            ) from e
        subfigures = []
        print("Producing subfigures")
        heights = []
        for encoder in self.keys():
            for record in tqdm(self[encoder], desc=f"{encoder}"):
                figure = record.plot_topics_with_images(
                    n_cols=n_cols,
                    grid_size=grid_size,
                    image_size=image_size,
                    scale_factor=scale_factor,
                )
                figure.update_layout(
                    title=encoder + " - " + figure.layout.title.text
                )
                heights.append(figure.layout.height)
                subfigures.append(figure)
        print("Joining figures")
        fig = subfigures[0]
        steps = []
        for i, sub in enumerate(subfigures):
            step = dict(
                method="relayout",
                label=sub.layout.title.text,
                args=[
                    {
                        "annotations": sub.layout.annotations,
                        "images": sub.layout.images,
                    },
                ],
            )
            steps.append(step)
        sliders = [
            dict(
                active=0,
                name="Switch between models",
                currentvalue={"prefix": "Entry: "},
                pad={"t": 20, "b": 20},
                steps=steps,
            )
        ]
        fig = fig.update_layout(
            sliders=sliders,
            margin=dict(l=0, r=0, b=0, t=0),
            title="",
            height=max(heights) + 20,
        )
        return fig
