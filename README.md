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

Creates a latex table of the results of the benchmark.

```bash
python3 -m topic_benchmark table -o results.tex
```

| argument | description | type | default |
| -------- | ----------- | ---- | ------- |
| results_folder | The folder where all result files are located. | `str` | `"results/"` |
| --out_file (-o) | The output path of the benchmark results. By default, results will be printed to stdout. | `str` | `None` | 

## Reproducing $S^3$ paper results
Result files to all runs in the $S^3$ publication can be found in the `results/` folder in the repository.
To reproduce the results reported in our paper, please do the following.

First, install this package by running the following command:

```bash
pip install topic-benchmark==0.3.0
```

Then, reproduce results for all the embedding models tested in the paper by running the following CLI commands:
```bash
python3 -m topic_benchmark run -e all-Mini-L6-v2
python3 -m topic_benchmark run -e all-mpnet-base-v2
python3 -m topic_benchmark run -e average_word_embeddings_glove.6B.300d
python3 -m topic_benchmark run -e intfloat/e5-large-v2
```

The results for each embedding model will be found in the `results` folder (unless a value for `--out_file` is explicitly passed).

To produce figures and tables in the paper, you can use the scripts in the  `s3_paper_scripts/` folder.
```bash
pip install -r s3_paper_scripts/requirements.txt

# Table 3: Main Table (tables/main_table.tex)
python3 s3_paper_scripts/main_table.py

# Figure 2: Preprocessing effects (figures/effect_of_preprocessing.png)
python3 s3_paper_scripts/effect_of_preprocessing.py

# Figure 3: Stop word frequency in topic descriptions (figures/stop_freq.png)
python3 s3_paper_scripts/stop_words_figure.py

# Table 4: Average percentage runtime difference from S^3 (tables/speed.tex)
python3 s3_paper_scripts/speed.py

# Table 5: Topics in ArXiv ML (tables/arxiv_ml_topics.tex)
# Figure 4: Compass of Concepts in ArXiv ML (figures/arxiv_ml_map.png)
python3 s3_paper_scripts/arxiv_ml_compass.py

##################
#### APPENDIX ####
##################

# Table 6: NPMI Coherence of topics (tables/npmi_table.tex)
python3 s3_paper_scripts/npmi_table.py

# Figures 5-9: Disaggregated results (figures/disaggregated_{metric_name}.png)
python3 s3_paper_scripts/disaggregated_results_figures.py
```

