import numpy as np
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from turftopic import SemanticSignalSeparation

from topic_benchmark.datasets.arxiv_ml import load_arxiv_ml

corpus = load_arxiv_ml

trf = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = trf.encode(corpus, show_progress_bar=True)

model = SemanticSignalSeparation(
    5, encoder=trf, vectorizer=CountVectorizer(), random_state=42
)
topic_data = model.prepare_topic_data(corpus, embeddings=embeddings)

model.print_topics(top_k=5)

print(model.export_topics(top_k=5, format="latex"))

topic_content = pd.DataFrame(
    topic_data["topic_term_matrix"].T, columns=model.topic_names
)
topic_content["word"] = topic_data["vocab"]
topic_content["freq"] = np.squeeze(
    np.asarray(topic_data["document_term_matrix"].sum(axis=0))
)
topic_content = topic_content[~topic_content["word"].isin(ENGLISH_STOP_WORDS)]

topic1 = 1
topic2 = 4
x = topic_content[model.topic_names[topic1]]
y = topic_content[model.topic_names[topic2]]
points = np.array(list(zip(x, y)))
xx, yy = np.meshgrid(np.arange(-3, 3, 0.32), np.arange(-3, 3, 0.32))
coords = np.array(list(zip(np.ravel(xx), np.ravel(yy))))
coords = coords + np.random.default_rng(0).normal(
    [0, 0], [0.1, 0.1], size=coords.shape
)
dist = euclidean_distances(coords, points)
idxs = np.argmin(dist, axis=1)
disp = topic_content.iloc[np.unique(idxs)]
fig = px.scatter(
    disp,
    x=model.topic_names[topic1],
    y=model.topic_names[topic2],
    text="word",
    template="plotly_white",
)
fig = fig.update_traces(
    mode="text", textfont_color="black", marker=dict(color="black")
).update_layout(
    xaxis_title=f"Topic {topic1}: Physical, Biological, Vision vs. Linguistic Problems",
    yaxis_title=f"Topic: {topic2}: Deep Learning vs. Algorithms",
)
fig = fig.update_layout(
    width=700,
    height=700,
    font=dict(family="Times New Roman", color="black", size=21),
    margin=dict(l=5, r=5, t=5, b=5),
)
fig = fig.add_hline(y=0, line_color="black", line_width=4)
fig = fig.add_vline(x=0, line_color="black", line_width=4)
fig = fig.update_xaxes(range=(-2.6, 1.75))
fig = fig.update_yaxes(range=(-1.5, 2.15))
fig.write_image("figures/arxiv_ml_map.png", scale=2)
fig.show()
