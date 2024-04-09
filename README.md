# topic-benchmark
Command Line Interface for benchmarking topic models.

The package contains `catalogue` registries for all models, datasets and metrics for model evaluation,
along with scripts for producing tables and figures for the S3 paper.

## Usage:

### Installation

You can install the package from PyPI.

```bash
pip install topic-benchmark

```

### Commands

#### `run`

Run the benchmark using a given embedding model.
Runs can be resumed if they get obruptly stopped from the results file.

```bash
python3 -m topic_benchmark run -e "embedding_model_name"
```

| argument | description | type | default |
| -------- | ----------- | ---- | ------- |
| --encoder_model (-e) | The encoder model to use for the benchmark. | `str` | `"all-MiniLM-L6-v2"` |
| --out_file (-o) | The output path of the benchmark results. By default it will be under `results/{encoder_model}.jsonl` | `str` | `None` | 

### `table`

Creates a latex table of the results of the benchmark. (Main table in the paper)

```bash
python3 -m topic_benchmark table -o results.tex
```

| argument | description | type | default |
| -------- | ----------- | ---- | ------- |
| results_folder | The folder where all result files are located. | `str` | `"results/"` |
| --out_file (-o) | The output path of the benchmark results. By default, results will be printed to stdout. | `str` | `None` | 

### `figures`

Creates all figures in the paper as `.png` files.

```bash
python3 -m topic_benchmark figures
```

| argument | description | type | default |
| -------- | ----------- | ---- | ------- |
| results_folder | The folder where all result files are located. | `str` | `"results/"` |
| --out_file (-o) | Directory where the figures should be placed.  | `str` | `"figures/"` | 
| --show_figures (-s) | Indicates whether the figures should be displayed in a browser tab or not. | `bool` | `False` | 
