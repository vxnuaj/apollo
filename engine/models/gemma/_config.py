import jax
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
from typing import List, Tuple, Dict, Any
from flax.linen.initializers import xavier_normal, zeros_init
from pydantic_core.core_schema import dataclass_schema
from flax.linen.linear import PromoteDtypeFn
from flax.typing import Initializer

@dataclass(config = ConfigDict(arbitrary_types_allowed = True))
class EmbeddingConfig:
    num_embeddings: int
    features: int
    dtype: jax.typing.DTypeLike
    param_dtype: jax.typing.DTypeLike
    embedding_init: Initializer

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