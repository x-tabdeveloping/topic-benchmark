import os

from topic_benchmark.cli import cli

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    cli.run()
