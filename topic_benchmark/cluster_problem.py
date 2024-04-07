import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_circles, make_moons

moons, _ = make_moons(n_samples=2000, noise=0.05)
circles, _ = make_circles(n_samples=3000, noise=0.01)


def show_clusters_with_centroids(data):
    clusters = HDBSCAN().fit_predict(data)
    centroids = []
    unique_labels = np.unique(clusters)
    unique_labels = np.sort(unique_labels)
    for label in unique_labels:
        centroid = np.mean(data[clusters == label], axis=0)
        centroids.append(centroid)
    centroids = np.stack(centroids)
    cluster_labels = [
        f"Cluster {i}" if i != -1 else "Outlier" for i in clusters
    ]
    colors = ["cornflowerblue", "indianred"]
    fig = px.scatter(
        x=data[:, 0],
        y=data[:, 1],
        color=cluster_labels,
        color_discrete_sequence=colors,
        template="plotly_white",
    )
    for i, centroid in enumerate(centroids):
        fig.add_annotation(
            text="<b>Centroid",
            x=centroid[0],
            y=centroid[1],
            showarrow=True,
            arrowhead=6,
            font=dict(size=36, color="white"),
            bgcolor=colors[i],
            # arrowcolor=colors[i],
            ax=-50 if i == 0 else 50,
            ay=-50 if i == 0 else 50,
            arrowsize=1.5,
            arrowwidth=2,
            bordercolor="black",
            borderwidth=3,
        )
    return fig


fig = make_subplots(rows=1, cols=2)
subplots = []
subplots.append(show_clusters_with_centroids(moons))
subplots.append(show_clusters_with_centroids(circles))
for col, subplot in enumerate(subplots):
    for trace in subplot.data:
        fig.add_trace(trace, col=col + 1, row=1)
    for annotation in subplot.layout.annotations:
        fig.add_annotation(annotation, col=col + 1, row=1)
fig = fig.update_layout(
    template="plotly_white",
    width=800,
    height=400,
    margin=dict(l=20, r=20, t=20, b=40),
    font_family="CMU Serif",
    showlegend=False,
)

fig.update_xaxes(showgrid=False, showticklabels=False, ticks="", showline=False, zeroline=False)
fig.update_yaxes(showgrid=False, showticklabels=False, ticks="", showline=False, zeroline=False)
fig.write_image("figures/cluster_centroid_problem.png", scale=3)
