from urllib.request import urlopen

from sklearn.datasets import fetch_20newsgroups

from topic_benchmark.registries import dataset_registry


@dataset_registry.register("20 Newsgroups Preprocessed")
def load_newsgroups_clean() -> list[str]:
    corpus_url = "https://raw.githubusercontent.com/MIND-Lab/OCTIS/master/preprocessed_datasets/20NewsGroup/corpus.tsv"
    corpus = []
    with urlopen(corpus_url) as in_file:
        corpus_text = in_file.read().decode("utf-8")
        lines = corpus_text.split("\n")
        for line in lines:
            text, *_ = line.split("\t")
            corpus.append(text)
    return corpus


@dataset_registry.register("20 Newsgroups Raw")
def load_newsgroups_raw() -> list[str]:
    ds = fetch_20newsgroups(subset="all")
    corpus = list(ds.data)
    return corpus
