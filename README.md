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

Run the benchmark.

```bash
python3 -m topic_benchmark run
```

| Option                   | Short Flag | Type                  | Optional | Description                                                                 | Default      |
|--------------------------|------------|-----------------------|----------|-----------------------------------------------------------------------------|---------------|
| `--help`                 | `-h`       | Flag (bool)           | Yes      | Show this help message and exit                                             | —             |
| `--out_dir OUT_DIR`      | `-o`       | `str`                 | Yes      | Output directory for the results                                            | `results/`    |
| `--encoders ENCODERS`    | `-e`       | `str`                 | Yes      | Which encoders should be used for conducting runs                           | `None`        |
| `--models MODELS`        | `-m`       | `list[str]` or `None` | Yes      | Subsection of models to benchmark                                           | `None`        |
| `--datasets DATASETS`    | `-d`       | `list[str]` or `None` | Yes      | Datasets to evaluate models on                                              | `None`        |
| `--metrics METRICS`      | `-t`       | `list[str]` or `None` | Yes      | Metrics to evaluate models on                                               | `None`        |
| `--seeds SEEDS`          | `-s`       | `list[int]` or `None` | Yes      | Seeds to evaluate models on                                                 | `None`        |
| `--multimodal`           | —          | `bool` (flag)         | Yes      | Indicates if the benchmark should be multimodal                             | `False`       |
| `--strict`               | —          | `bool` (flag)         | Yes      | Indicates if the benchmark should fail on error                             | `False`       |

### Push to hub

Push results to a HuggingFace repository.

```bash
python3 -m topic_benchmark push_to_hub "your_user/your_repo"
```

| Argument          | Description                                            | Type  | Default    |
|-------------------|--------------------------------------------------------|-------|------------|
| `hf_repo`         | HuggingFace repository to push results to.             | `str` | N/A        |
| `results_folder`  | Folder containing results for all embedding models.    | `str` | `results/` |

## Reproducing $S^3$ paper results
Result files to all runs in the $S^3$ publication can be found in the `results/` folder in the repository.
To reproduce the results reported in our paper, please do the following.

First, install this package by running the following command:

> Note: We used an older version of the package for the $S^3$ paper, and have introduced breaking changes since then. Please use version 0.6.0 if you intend to get the same result format.

```bash
pip install topic-benchmark==0.6.0
python3 -m topic-benchmark run -o results/
```

The results for each embedding model will be found in the `results` folder (unless a value for `--out_file` is explicitly passed).

To produce figures and tables in the paper, you can use the scripts in the  `scripts/s3_paper/` folder.
