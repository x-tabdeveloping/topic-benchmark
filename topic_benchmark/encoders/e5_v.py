from __future__ import annotations

from typing import Any

import numpy as np
import torch
import transformers
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from turftopic.encoders.multimodal import MultimodalEncoder

from topic_benchmark.registries import encoder_registry


class E5VWrapper(MultimodalEncoder):
    def __init__(
        self,
        model_name: str,
        composed_prompt=None,
        **kwargs: Any,
    ):

        self.model_name = model_name
        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        if "device" in kwargs:
            self.device = kwargs.pop("device")
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name, **kwargs
        )
        self.model.eval()
        self.template = "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n"
        self.text_prompt = self.template.format(
            "<sent>\nSummary above sentence in one word: "
        )
        self.img_prompt = self.template.format(
            "<image>\nSummary above image in one word: "
        )
        if not composed_prompt:
            # default composed embedding, to_do: move it to get_fused_embedding with "prompt_name" like MTEB text ones.
            self.composed_prompt = self.template.format(
                '[INST] <image> Modify this image with "{}" Describe modified image in one word: [/INST]'
            )
        else:
            self.composed_prompt = self.template.format(composed_prompt)

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str,
        prompt_type,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.get_text_embeddings(
            sentences, task_name=task_name, prompt_type=prompt_type
        )

    def get_text_embeddings(
        self,
        texts: list[str],
        *,
        task_name: str | None = None,
        prompt_type,
        batch_size: int = 8,
        **kwargs: Any,
    ):
        all_text_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i : i + batch_size]
                text_inputs = self.processor(
                    [
                        self.text_prompt.replace("<sent>", text)
                        for text in batch_texts
                    ],
                    return_tensors="pt",
                    padding=True,
                ).to("cuda")
                text_outputs = self.model(
                    **text_inputs, output_hidden_states=True, return_dict=True
                ).hidden_states[-1][:, -1, :]
                all_text_embeddings.append(text_outputs.cpu())
        return torch.cat(all_text_embeddings, dim=0)

    def get_image_embeddings(
        self,
        images: list[Image.Image] | DataLoader,
        *,
        task_name: str | None = None,
        prompt_type,
        batch_size: int = 8,
        **kwargs: Any,
    ):
        all_image_embeddings = []

        with torch.no_grad():
            if isinstance(images, DataLoader):
                for batch_images in tqdm(images):
                    img_inputs = self.processor(
                        [self.img_prompt] * len(batch_images),
                        batch_images,
                        return_tensors="pt",
                        padding=True,
                    ).to("cuda")
                    image_outputs = self.model(
                        **img_inputs,
                        output_hidden_states=True,
                        return_dict=True,
                    ).hidden_states[-1][:, -1, :]
                    all_image_embeddings.append(image_outputs.cpu())
            else:
                for i in tqdm(range(0, len(images), batch_size)):
                    batch_images = images[i : i + batch_size]
                    img_inputs = self.processor(
                        [self.img_prompt] * len(batch_images),
                        batch_images,
                        return_tensors="pt",
                        padding=True,
                    ).to("cuda")
                    image_outputs = self.model(
                        **img_inputs,
                        output_hidden_states=True,
                        return_dict=True,
                    ).hidden_states[-1][:, -1, :]
                    all_image_embeddings.append(image_outputs.cpu())
            return torch.cat(all_image_embeddings, dim=0)

    def calculate_probs(self, text_embeddings, image_embeddings):
        text_embeddings = text_embeddings / text_embeddings.norm(
            dim=-1, keepdim=True
        )
        image_embeddings = image_embeddings / image_embeddings.norm(
            dim=-1, keepdim=True
        )
        logits = torch.matmul(image_embeddings, text_embeddings.T)
        probs = (logits * 100).softmax(dim=-1)
        return probs

    def get_fused_embeddings(
        self,
        texts: list[str] = None,
        images: list[Image.Image] = None,
        batch_size: int = 8,
        **kwargs: Any,
    ):
        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")

        all_fused_embeddings = []
        kwargs.update(batch_size=batch_size)

        if texts is not None and images is not None:
            with torch.no_grad():
                if isinstance(images, DataLoader):
                    for index, batch_images in enumerate(tqdm(images)):
                        batch_texts = texts[
                            index * batch_size : (index + 1) * batch_size
                        ]
                        prompts = [
                            self.composed_prompt.format(text)
                            for text in batch_texts
                        ]
                        inputs = self.processor(
                            prompts,
                            batch_images,
                            return_tensors="pt",
                            padding=True,
                        ).to("cuda")
                        outputs = self.model(
                            **inputs,
                            output_hidden_states=True,
                            return_dict=True,
                        ).hidden_states[-1][:, -1, :]
                        all_fused_embeddings.append(outputs.cpu())
                else:
                    if len(texts) != len(images):
                        raise ValueError(
                            "The number of texts and images must have the same length"
                        )
                    for i in tqdm(range(0, len(images), batch_size)):
                        batch_texts = texts[i : i + batch_size]
                        batch_images = images[i : i + batch_size]
                        prompts = [
                            self.composed_prompt.format(text)
                            for text in batch_texts
                        ]
                        inputs = self.processor(
                            prompts,
                            batch_images,
                            return_tensors="pt",
                            padding=True,
                        ).to("cuda")
                        outputs = self.model(
                            **inputs,
                            output_hidden_states=True,
                            return_dict=True,
                        ).hidden_states[-1][:, -1, :]
                        all_fused_embeddings.append(outputs.cpu())
            return torch.cat(all_fused_embeddings, dim=0)
        elif texts is not None:
            text_embeddings = self.get_text_embeddings(texts, **kwargs)
            return text_embeddings
        elif images is not None:
            image_embeddings = self.get_image_embeddings(images, **kwargs)
            return image_embeddings


@encoder_registry.register("royokong/e5-v")
def create_e5_v() -> MultimodalEncoder:
    return E5VWrapper("royokong/e5-v")
