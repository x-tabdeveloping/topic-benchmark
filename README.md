# topic-benchmark
Command Line Interface for benchmarking topic models.

The package contains `catalogue` registries for all models, datasets and metrics for model evaluation,
along with scripts for producing tables and figures for the S3 paper.

## Usage

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

## Reproducing paper results
To reproduce results reported in our paper, please do the following.

First, install this package by running the following CLI command:

```bash
pip install topic-benchmark
```

Then, reproduce results for all the embedding models tested in the paper by running the following CLI commands:
```bash
python3 -m topic_benchmark run -e all-Mini-L6-v2
python3 -m topic_benchmark run -e all-mpnet-base-v2
python3 -m topic_benchmark run -e average_word_embeddings_glove.6B.300d
python3 -m topic_benchmark run -e intfloat/e5-large-v2
```

The results for each embedding model will be found in the `results` folder (unless a value for `--out_file` is explicitly passed).

Each of these commands will compute results for all metrics, all datasets, all topic modeling methods, and all hyperparameters (i.e., number of topics) reported in the paper given a specific embedding model. Note that running these commends can be therefore be very time consuming, due to the large number of models that are estimated and evaluated.

To reproduce the result table reported in the paper, please run: 

```bash
python3 -m topic_benchmark table -o results.tex
```

To reproduce the figures included in the paper, please run:
```bash
python3 -m topic_benchmark figures
```
