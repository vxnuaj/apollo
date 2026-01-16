import jax.numpy as jnp
import jax
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict, field_validator
from typing import List, Tuple, Dict, Any, Union
from flax.linen.initializers import xavier_normal, zeros_init
from pydantic_core.core_schema import dataclass_schema
from flax.linen.linear import PromoteDtypeFn
from flax.typing import Initializer

@dataclass(config = ConfigDict(arbitrary_types_allowed = True))
class EmbeddingConfig:
    num_embeddings: int
    features: int
    dtype:Union[str, jax.typing.DTypeLike] # fp32, fp16, bf16, etc
    param_dtype: Union[str, jax.typing.DTypeLike] # fp32, fp16, bf16, etc
    embedding_init: Initializer

    @field_validator("dtype", "param_dtype", mode = "before")
    @classmethod
    def validate_dtypes(cls, v):
        valid_dtypes = ["fp32", "fp16", "bf16"]
        if v not in valid_dtypes: raise ValueError(f"{v} is not a valid dtype, expected fp32, fp16, or bf16")
        if v == "fp32": return jnp.float32
        if v == "fp16": return jnp.float16
        if v == "bf16": return jnp.bfloat16

@dataclass(config = ConfigDict(arbitrary_types_allowed = True))
class AttentionConfig:
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    rope_base_frequency: int = 10_000
    rope_scale_factor: float = 1.0
    rope_proportion: float = 1.0
    
@dataclass(config = ConfigDict(arbitrary_types_allowed = True))
class TransformerBlockConfig:
    attn_config: AttentionConfig
    ffn_hidden_dim: int
    embed_dim: int

@dataclass(config = ConfigDict(arbitrary_types_allowed = True))
class GemmaConfig:
    embedding_config: EmbeddingConfig
    transformer_block_config: TransformerBlockConfig
    num_layers: int