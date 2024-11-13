"""
Registry of E5 models
"""

from sentence_transformers import SentenceTransformer

from topic_benchmark.registries import encoder_registry

e5_prompts = {"query": "query: ", "passage": "passage: "}


@encoder_registry.register("intfloat/e5-large-v2")
def create_e5_large_v2() -> SentenceTransformer:
    hf_name = "intfloat/e5-large-v2"
    return SentenceTransformer(
        hf_name, prompts=e5_prompts, default_prompt_name="query"
    )


@encoder_registry.register("intfloat/multilingual-e5-small")
def create_m_e5_small() -> SentenceTransformer:
    hf_name = "intfloat/multilingual-e5-small"
    return SentenceTransformer(
        hf_name, prompts=e5_prompts, default_prompt_name="query"
    )
