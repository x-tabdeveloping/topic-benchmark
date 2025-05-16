import base64
from collections import namedtuple
from io import BytesIO
from typing import Any, Optional, Type

import msgspec
import numpy as np
from PIL import Image
from turftopic.container import TopicContainer

EntryID = namedtuple("EntryID", ["dataset", "model", "n_topics", "seed"])


def image_enc_hook(obj: Any) -> Any:
    if isinstance(obj, Image.Image):
        # Convert image to base64 encoding
        buffer = BytesIO()
        obj.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    if isinstance(obj, np.str_):
        return str(obj)
    else:
        # Raise a NotImplementedError for other types
        raise NotImplementedError(
            f"Objects of type {type(obj)} are not supported"
        )


def image_dec_hook(type: Type, obj: Any) -> Any:
    # `type` here is the value of the custom type annotation being decoded.
    if type is Image.Image:
        buffer = BytesIO(base64.b64decode(obj))
        img = Image.open(buffer)
        return img
    else:
        # Raise a NotImplementedError for other types
        raise NotImplementedError(f"Objects of type {type} are not supported")


class BenchmarkEntry(msgspec.Struct):
    dataset: str
    model: str
    n_topics: int
    seed: int
    topic_descriptions: list[list[str]]
    runtime_s: float
    results: dict[str, float]
    top_documents: Optional[list[list[str]]] = None
    top_images: Optional[list[list[Image.Image]]] = None
    error_message: Optional[str] = None

    @property
    def entry_id(self) -> EntryID:
        return EntryID(
            self.dataset,
            self.model,
            self.n_topics,
            self.seed,
        )

    @classmethod
    def error(
        cls,
        dataset: str,
        model: str,
        n_topics: int,
        seed: int,
        error_message: str,
    ):
        return cls(
            dataset=dataset,
            model=model,
            n_topics=n_topics,
            seed=seed,
            topic_descriptions=[],
            top_documents=[],
            top_images=[],
            runtime_s=-1,
            results={},
            error_message=error_message,
        )

    @staticmethod
    def encoder():
        return msgspec.json.Encoder(enc_hook=image_enc_hook)

    @classmethod
    def decoder(cls):
        return msgspec.json.Decoder(cls, dec_hook=image_dec_hook)

    def to_json(self):
        return self.encoder().encode(self).decode("utf-8")

    @classmethod
    def from_json(cls, json_str: str):
        return cls.decoder().decode(json_str.encode("utf-8"))

    def to_dict(self) -> dict:
        return dict(
            dataset=self.dataset,
            model=self.model,
            n_topics=self.n_topics,
            seed=self.seed,
            topic_descriptions=self.topic_descriptions,
            top_documents=self.top_documents,
            top_images=self.top_images,
            runtime_s=self.runtime_s,
            results=self.results,
            error_message=self.error_message,
        )

    def plot_topics_with_images(
        self,
        n_cols: int = 6,
        grid_size: int = 4,
        image_size: int = 1200,
        scale_factor: float = 0.25,
    ):
        """Plots the most important images for each topic, along with keywords."""
        if self.top_images is None:
            raise TypeError(
                "Can't display images for a result object that does not have them."
            )
        try:
            import plotly.graph_objects as go
        except (ImportError, ModuleNotFoundError) as e:
            raise ModuleNotFoundError(
                "Please install plotly if you intend to use plots in Turftopic."
            ) from e
        title = f"{self.model}({self.n_topics}) - {self.dataset}"
        fig = go.Figure()
        width, height = image_size, image_size
        w, h = width * scale_factor, height * scale_factor
        padding = 10
        topics = self.topic_descriptions
        n_components = len(topics)
        n_rows = n_components // n_cols + int(bool(n_components % n_cols))
        figure_height = (h + padding) * n_rows
        figure_width = (w + padding) * n_cols
        fig = fig.add_trace(
            go.Scatter(
                x=[0, figure_width],
                y=[0, figure_height],
                mode="markers",
                marker_opacity=0,
            )
        )
        for i, topic in enumerate(topics):
            col = i % n_cols
            row = i // n_cols
            images = self.top_images[i]
            image = TopicContainer._image_grid(
                images, (width, height), grid_size=(grid_size, grid_size)
            )
            x0 = (w + padding) * col
            y0 = (h + padding) * (n_rows - row)
            fig = fig.add_layout_image(
                dict(
                    x=x0,
                    sizex=w,
                    y=y0,
                    sizey=h,
                    xref="x",
                    yref="y",
                    opacity=1.0,
                    layer="below",
                    sizing="stretch",
                    source=image,
                ),
            )
            fig.add_annotation(
                x=(w + padding) * col + (w / 2),
                y=(h) * (n_rows - row) - (h / 2),
                text="<b> " + "<br> ".join(topic),
                font=dict(
                    size=16,
                    family="Times New Roman",
                    color="white",
                ),
                bgcolor="rgba(0,0,0, 0.5)",
            )
        fig = fig.update_xaxes(visible=False, range=[0, figure_width])
        fig = fig.update_yaxes(
            visible=False,
            range=[0, figure_height],
            # the scaleanchor attribute ensures that the aspect ratio stays constant
            scaleanchor="x",
        )
        fig = fig.update_layout(
            width=figure_width,
            height=figure_height,
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
            title=dict(text=title, font=dict(size=20)),
        )
        return fig
