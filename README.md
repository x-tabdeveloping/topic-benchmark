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

Run the benchmark. Defaults to running all models with the benchmark used in Kardos et al. (2024).

```bash
python3 -m topic_benchmark run
```

| Argument               | Short Flag | Description                                                                                              | Type                                   | Default         |
|------------------------|------------|----------------------------------------------------------------------------------------------------------|----------------------------------------|-----------------|
| `--out_dir OUT_DIR`    | `-o`       | Output directory for the results.                                                                        | `str`                                  | `results/`      |
| `--encoders ENCODERS`  | `-e`       | Which encoders should be used for conducting runs?                                                       | `str`                                  | `None`          |
| `--models MODELS`      | `-m`       | What subsection of models should the benchmark be run on.                                                | `Optional[list[str], NoneType]`        | `None`          |
| `--datasets DATASETS`  | `-d`       | What datasets should the models be evaluated on.                                                         | `Optional[list[str], NoneType]`        | `None`          |
| `--metrics METRICS`    | `-t`       | What metrics should the models be evaluated on.                                                          | `Optional[list[str], NoneType]`        | `None`          |
| `--seeds SEEDS`        | `-s`       | What seeds should the models be evaluated on.                                                            | `Optional[list[int], NoneType]`        | `None`          |


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

```bash
pip install topic-benchmark
python3 -m topic-benchmark run -o results/
```

The results for each embedding model will be found in the `results` folder (unless a value for `--out_file` is explicitly passed).

To produce figures and tables in the paper, you can use the scripts in the  `scripts/s3_paper/` folder.
