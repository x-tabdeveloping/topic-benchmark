"""
Registry of E5 models
"""

from turftopic.encoders import E5Encoder

from topic_benchmark.registries import encoder_registry


@encoder_registry.register("intfloat/e5-large-v2")
def create_e5_large_v2() -> E5Encoder:
    hf_name = "intfloat/e5-large-v2"
    return E5Encoder(model_name=hf_name, prefix="query: ")


@encoder_registry.register("intfloat/multilingual-e5-small")
def create_m_e5_small() -> E5Encoder:
    hf_name = "intfloat/multilingual-e5-small"
    return E5Encoder(model_name=hf_name, prefix="query: ")


@encoder_registry.register("intfloat/multilingual-e5-large-instruct")
def create_m_e5_large_instruct() -> E5Encoder:
    hf_name = "intfloat/multilingual-e5-large-instruct"
    task_description = "" #TODO
    prefix = f"Instruct: {task_description} \nQuery: "
    return E5Encoder(model_name=hf_name, prefix=prefix)
