import numpy as np
from turftopic.data import TopicData


def get_top_k(data: TopicData, top_k: int = 10) -> list[list[str]]:
    topic_names = data["topic_names"]
    components = data["topic_term_matrix"]
    vocab = data["vocab"]
    n_topics = components.shape[0]
    if not len(topic_names):
        topic_names = [" " for _ in range(n_topics)]
    res = []
    for name, component in zip(topic_names, components):
        # Skipping outlier topics
        if name.startswith("-1"):
            continue
        high = np.argpartition(-component, top_k)[:top_k]
        res.append(list(vocab[high]))
    return res
